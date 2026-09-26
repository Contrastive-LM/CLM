#!/usr/bin/env python3
"""Prepare tutorial inputs and assemble their embeddings without changing the trainer."""

import argparse
import array
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import zlib

from encoding import atomic_json, batch_digest, batch_rows, load_batch

class ChunkStore:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute('CREATE TABLE chunks (id INTEGER PRIMARY KEY, digest TEXT UNIQUE NOT NULL, tokens BLOB NOT NULL, length INTEGER NOT NULL)')
        self.ids = {}
        self.total_tokens = 0

    def add(self, tokens):
        if not tokens:
            raise ValueError('Empty chunk')
        raw = array.array('I', tokens).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        return self.add_encoded(digest, zlib.compress(raw), len(tokens))

    def add_encoded(self, digest, compressed, length):
        if digest not in self.ids:
            index = len(self.ids)
            self.connection.execute('INSERT INTO chunks VALUES (?, ?, ?, ?)', (index, digest, compressed, length))
            self.ids[digest] = index
            self.total_tokens += length
        return self.ids[digest]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def initialize(source, model):
    global recipe
    sys.path.insert(0, str(Path(source) / 'train'))
    from embed_utils import Recipe
    recipe = Recipe(model, 8192)


def tokenize(row):
    """Call the pinned release recipe on each complete causal message prefix."""
    history, steps, encoded = [], [], {}

    def add(ids):
        if not 0 < len(ids) <= 8191:
            raise ValueError('Invalid tutorial input length')
        raw = array.array('I', ids).tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        encoded.setdefault(digest, (zlib.compress(raw), len(ids)))
        return digest

    for message in row['messages']:
        if message['role'] == 'assistant':
            steps.append(dict(state=add(recipe.state_ids(history)),
                              action=add(recipe.text_ids(message['content'], keep='head'))))
        history.append(message)
    if not steps or history[-1]['role'] != 'assistant':
        raise ValueError('Incomplete conversation')
    return {k: v for k, v in row.items() if k != 'messages'}, steps, encoded


def validate_cohort(cohort):
    families, tasks, traces = {}, {}, set()
    for split, rows in cohort.items():
        if split not in {'train', 'val', 'test'}:
            raise ValueError('Unknown split')
        for row in rows:
            if row['trace_id'] in traces:
                raise ValueError('Duplicate trace')
            traces.add(row['trace_id'])
            if families.setdefault(row['task_group'], split) != split:
                raise ValueError('Task family crosses outer splits')
            if tasks.setdefault(row['task_id'], split) != split:
                raise ValueError('Task crosses outer splits')
            if split == 'train' and not row['passed']:
                raise ValueError('Tutorial correspondence training requires successful traces')


def prepare(args, plan):
    cohort = json.loads(Path(plan['cohort']).read_text())
    validate_cohort(cohort)
    expected = {r['trace_id']: r for rows in cohort.values() for r in rows}
    destination = args.data / 'tokenized'
    destination.mkdir(exist_ok=False)
    store = ChunkStore(destination / 'chunks.sqlite')
    seen, counts, token_counts = set(), Counter(), Counter()
    output_path = destination / 'traces.jsonl'
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(plan['source'], args.model)) as pool:
        with gzip.open(plan['conversations'], 'rt') as stream, output_path.open('x') as output:
            pending = deque()
            exhausted = False
            while pending or not exhausted:
                while len(pending) < 2 * args.workers and not exhausted:
                    line = stream.readline()
                    if not line:
                        exhausted = True
                        break
                    row = json.loads(line)
                    reference = expected[row['trace_id']]
                    for key in ('task_id', 'task_group', 'source', 'split', 'passed', 'content_digest'):
                        if row[key] != reference[key]:
                            raise ValueError(f'Conversation differs from cohort: {key}')
                    if row['trace_id'] in seen:
                        raise ValueError('Repeated conversation')
                    seen.add(row['trace_id'])
                    pending.append(pool.submit(tokenize, row))
                if not pending:
                    continue
                row, steps, encoded = pending.popleft().result()
                mapping = {digest: store.add_encoded(digest, blob, length)
                           for digest, (blob, length) in encoded.items()}
                for step in steps:
                    for side in ('state', 'action'):
                        token_counts[side] += encoded[step[side]][1]
                        step[side] = mapping[step[side]]
                if len(steps) != expected[row['trace_id']]['steps']:
                    raise ValueError('Changed step coverage')
                output.write(json.dumps(dict(row, steps=steps)) + '\n')
                counts[row['split'] + '_traces'] += 1
                counts[row['split'] + '_steps'] += len(steps)
                if sum(counts[k] for k in counts if k.endswith('_traces')) % 25 == 0:
                    store.connection.commit()
                    print(json.dumps(dict(counts=counts, unique_tokens=store.total_tokens)), flush=True)
    if seen != set(expected):
        raise ValueError('Missing cohort conversations')
    store.connection.commit()
    store.connection.close()
    manifest = dict(model='Qwen/Qwen3-8B', revision=plan['encoder_revision'], chunk_tokens=8191,
                    recipe='release Recipe.state_ids and Recipe.text_ids(keep=head)',
                    state_context='last 8191 tokens after the Qwen chat template',
                    action_context='first 8191 tokens without special tokens',
                    pooling='LAST', normalization='l2', unique_chunks=len(store.ids),
                    unique_tokens=store.total_tokens, encoded_tokens_by_side=token_counts,
                    counts=counts, trace_plan_sha256=sha(output_path))
    atomic_json(destination / 'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


def encode(args, plan):
    from encoding import encode as encode_shard
    token_root = args.data / 'tokenized'
    manifest = json.loads((token_root / 'manifest.json').read_text())
    if sha(token_root / 'traces.jsonl') != manifest['trace_plan_sha256']:
        raise ValueError('Token plan changed')
    destination = args.data / 'encoded'
    (destination / 'batches').mkdir(exist_ok=True, parents=True)
    with sqlite3.connect(f'file:{token_root / "chunks.sqlite"}?mode=ro', uri=True) as db:
        if db.execute('SELECT count(*) FROM chunks').fetchone()[0] != manifest['unique_chunks']:
            raise ValueError('Incomplete token database')
        encode_shard(args, manifest, sha(token_root / 'manifest.json'), db, destination)


def assemble(args, plan):
    import torch
    torch.set_num_threads(1)
    token_root = args.data / 'tokenized'
    manifest = json.loads((token_root / 'manifest.json').read_text())
    if sha(token_root / 'traces.jsonl') != manifest['trace_plan_sha256']:
        raise ValueError('Token plan changed')
    for shard in range(plan['encoder_shards']):
        progress = json.loads((args.data / 'encoded/progress' / f'shard_{shard}.json').read_text())
        if progress['status'] != 'COMPLETED' or progress['tokens'] != progress['total_tokens']:
            raise ValueError('Encoder shard is incomplete')
    total = manifest['unique_chunks']
    vectors = torch.empty((total, 4096), dtype=torch.float16)
    digest = sha(token_root / 'manifest.json')
    with sqlite3.connect(f'file:{token_root / "chunks.sqlite"}?mode=ro', uri=True) as db:
        for start in range(0, total, 64):
            rows = batch_rows(db, start, total)
            path = args.data / 'encoded/batches' / f'{start:08d}.pt'
            vectors[start:start + len(rows)] = load_batch(path, batch_digest(digest, rows), len(rows))
    datasets = {name: [] for name in ('parthenon_cafl', 'val', 'test')}
    with (token_root / 'traces.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            names = ['parthenon_cafl'] if row['split'] == 'train' else [row['split']]
            for index, step in enumerate(row['steps']):
                sample = dict(task_id=row['task_id'], step_idx=index, trajectory_id=row['trace_id'],
                              reward=int(row['passed']), config='all_recorded_candidates',
                              task_group=row['task_group'], source=row['source'])
                for name in names:
                    datasets[name].append((step['state'], step['action'], sample))
    inventory = {}
    for name, rows in datasets.items():
        destination = args.data / 'reencoded_embeddings' / name
        destination.mkdir(parents=True, exist_ok=False)
        for column, side in enumerate(('state', 'action')):
            ids = torch.tensor([r[column] for r in rows])
            torch.save(vectors[ids], destination / f'{side}_embeddings.pt')
        samples = [r[2] for r in rows]
        atomic_json(destination / 'metadata.json', dict(samples=samples, num_samples=len(samples),
                    hidden_size=4096, model_name='Qwen/Qwen3-8B', max_model_len=8192))
        inventory[name] = dict(steps=len(samples), tasks=len({r['task_id'] for r in samples}),
                               traces=len({r['trajectory_id'] for r in samples}),
                               families=len({r['task_group'] for r in samples}))
    atomic_json(args.data / 'embedding_inventory.json', inventory)
    print(json.dumps(inventory), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    commands = parser.add_subparsers(dest='stage', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--workers', type=int, default=8)
    worker = commands.add_parser('encode')
    worker.add_argument('--shard', type=int, required=True)
    worker.add_argument('--num-shards', type=int, default=4)
    worker.add_argument('--max-model-len', type=int, default=8192)
    worker.add_argument('--max-num-seqs', type=int, default=8)
    worker.add_argument('--max-num-batched-tokens', type=int, default=32768)
    commands.add_parser('assemble')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    revision = 'b968826d9c46dd6066d109eabc6255188de91218'
    encoder = json.loads((args.data / 'encoder.json').read_text())
    if encoder['revision'] != revision:
        raise ValueError('Unexpected encoder revision')
    plan = dict(cohort=str(args.data / 'cohort.json'),
                conversations=str(args.data / 'conversations.jsonl.gz'),
                source=str(Path(__file__).resolve().parents[2]),
                encoder_revision=revision)
    if args.stage == 'prepare':
        args.model = snapshot_download('Qwen/Qwen3-8B', revision=revision,
                                      allow_patterns=['*.json', '*.txt', '*.jinja'])
    elif args.stage == 'encode':
        snapshot_download('Qwen/Qwen3-8B', revision=revision)
    else:
        settings = json.loads((args.data / 'encoded/sharding.json').read_text())
        plan['encoder_shards'] = settings['num_shards']
    globals()[args.stage](args, plan)


if __name__ == '__main__':
    main()
