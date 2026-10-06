#!/usr/bin/env python3
"""Encode disjoint complete-history shards, then assemble their shared cache."""

import argparse
import array
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import zlib



def atomic_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def shard_starts(chunks, shard, num_shards):
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError('Invalid shard assignment')
    return range(64 * shard, chunks, 64 * num_shards)


def batch_rows(db, start, total):
    rows = db.execute('SELECT id,digest,tokens,length FROM chunks WHERE id>=? AND id<? ORDER BY id',
                      (start, start + 64)).fetchall()
    if [r[0] for r in rows] != list(range(start, min(start + 64, total))):
        raise ValueError('Non-contiguous token indices')
    return rows


def batch_digest(manifest_digest, rows):
    return hashlib.sha256((manifest_digest + ''.join(r[1] for r in rows)).encode()).hexdigest()


def load_batch(path, digest, count):
    import torch
    saved = torch.load(path, weights_only=True, map_location='cpu')
    if set(saved) != {'digest', 'embeddings'} or saved['digest'] != digest:
        raise ValueError(f'Encoder batch does not match input: {path}')
    validate_vectors(saved['embeddings'], count)
    return saved['embeddings']


def validate_vectors(vectors, count):
    import torch
    if vectors.shape != (count, 4096) or not torch.isfinite(vectors).all() or (vectors.norm(dim=1) == 0).any():
        raise ValueError('Invalid chunk embeddings')


def local_model_snapshot(manifest):
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(manifest['model'], revision=manifest['revision'], local_files_only=True))
    if path.name != manifest['revision']:
        raise ValueError('Cached model snapshot does not match the pinned revision')
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    required = {'config.json', 'tokenizer.json', *index['weight_map'].values()}
    if any(not (path / name).is_file() for name in required):
        raise ValueError('Pinned model snapshot is incomplete')
    return str(path)


def encode(args, manifest, digest, db, destination):
    # The launcher downloads the pinned snapshot first. Inference needs no Hub requests.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import torch.nn.functional as F
    from vllm import LLM
    from vllm.config import PoolerConfig
    from vllm.inputs import TokensPrompt
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Each encoder process requires one visible GPU')
    progress_dir = destination / 'progress'
    progress_dir.mkdir(exist_ok=True)
    # One process owns each shard. Existing completed batches remain immutable.
    with (destination / 'sharding.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        settings = dict(num_shards=args.num_shards, manifest_digest=digest,
                        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                        max_num_batched_tokens=args.max_num_batched_tokens)
        path = destination / 'sharding.json'
        if path.exists() and json.loads(path.read_text()) != settings:
            raise ValueError('Shard plan differs from the active encoder run')
        atomic_json(path, settings)
    with (progress_dir / f'shard_{args.shard}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        total = manifest['unique_chunks']
        starts = list(shard_starts(total, args.shard, args.num_shards))
        sizes = dict(db.execute('SELECT id,length FROM chunks'))
        shard_chunks = sum(min(64, total - start) for start in starts)
        shard_tokens = sum(sizes[i] for start in starts for i in range(start, min(start + 64, total)))
        llm, done_chunks, done_tokens, new_tokens = None, 0, 0, 0
        started = time.monotonic()
        for start in starts:
            rows = batch_rows(db, start, total)
            identity = batch_digest(digest, rows)
            path = destination / 'batches' / f'{start:08d}.pt'
            if path.exists():
                load_batch(path, identity, len(rows))
            else:
                if llm is None:
                    snapshot = local_model_snapshot(manifest)
                    llm = LLM(model=snapshot, tokenizer=snapshot, revision=manifest['revision'], runner='pooling', convert='embed',
                              pooler_config=PoolerConfig(seq_pooling_type='LAST', use_activation=True),
                              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                              max_num_batched_tokens=args.max_num_batched_tokens,
                              gpu_memory_utilization=0.9, enable_prefix_caching=False, enable_chunked_prefill=False,
                              enforce_eager=True, dtype='bfloat16')
                    started = time.monotonic()
                prompts = []
                for _, expected, compressed, length in rows:
                    raw = zlib.decompress(compressed)
                    if hashlib.sha256(raw).hexdigest() != expected:
                        raise ValueError('Corrupt token chunk')
                    ids = array.array('I')
                    ids.frombytes(raw)
                    if len(ids) != length or length > manifest['chunk_tokens']:
                        raise ValueError('Invalid chunk length')
                    prompts.append(TokensPrompt(prompt_token_ids=ids.tolist()))
                outputs = llm.embed(prompts, use_tqdm=False)
                vectors = F.normalize(torch.tensor([o.outputs.embedding for o in outputs]), dim=1).half()
                validate_vectors(vectors, len(rows))
                temporary = path.with_suffix('.partial')
                torch.save(dict(digest=identity, embeddings=vectors), temporary)
                temporary.replace(path)
                new_tokens += sum(r[3] for r in rows)
            done_chunks += len(rows)
            done_tokens += sum(r[3] for r in rows)
            rate = new_tokens / max(time.monotonic() - started, 1)
            progress = dict(status='ENCODING', shard=args.shard, num_shards=args.num_shards,
                            chunks=done_chunks, total_chunks=shard_chunks, tokens=done_tokens, total_tokens=shard_tokens,
                            tokens_per_second=rate, remaining_seconds=(shard_tokens - done_tokens) / rate if rate else None)
            atomic_json(progress_dir / f'shard_{args.shard}.json', progress)
            print(json.dumps(progress), flush=True)
        progress.update(status='COMPLETED', remaining_seconds=0)
        atomic_json(progress_dir / f'shard_{args.shard}.json', progress)
        print(json.dumps(progress), flush=True)

