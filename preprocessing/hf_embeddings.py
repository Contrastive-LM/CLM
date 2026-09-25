#!/usr/bin/env python3
"""Convert between embedding dirs and Hugging Face parquet datasets.

An embedding dir (as written by the DeepSWE precompute in the research repo)::

    state_embeddings.pt    [N, H] tensor
    action_embeddings.pt   [N, H] tensor
    metadata.json          {"num_samples": N, "hidden_size": H,
                            "samples": [{"trajectory_id", "step_idx", "task_id",
                                         "reward", "model", "config"}, ...]}

Parquet rows are samples in order: the metadata columns plus ``state_embedding`` /
``action_embedding`` fixed-size lists.

    python preprocessing/hf_embeddings.py download Contrastive-LM/deepswe-clm-train-embeddings-8k \\
        --out data/deepswe_train
    python preprocessing/hf_embeddings.py export embeddings/train_pool/merged \\
        --out hf/deepswe_train --shards 16 --push <user>/<dataset>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

META_KEYS = ("trajectory_id", "step_idx", "task_id", "reward", "model", "config")
EMB_COLS = ("state_embedding", "action_embedding")
EMB_FILES = ("state_embeddings.pt", "action_embeddings.pt")
DEEPSWE_TRAIN_DATASET = "Contrastive-LM/deepswe-clm-train-embeddings-8k"
DEEPSWE_EVAL_DATASET = "Contrastive-LM/deepswe-clm-embeddings-8k"
TRAIN_DATASETS = {"deepswe": DEEPSWE_TRAIN_DATASET}
TRAIN_SPLITS = {"deepswe": None}


def is_embedding_dir(path: str) -> bool:
    """True when ``path`` holds a complete embedding dir."""
    return all(os.path.exists(os.path.join(path, f)) for f in (*EMB_FILES, "metadata.json"))


def _source_root(source: str, cache_dir: str | None) -> str:
    if os.path.isfile(source):
        return source
    if os.path.isdir(source):
        return source
    from huggingface_hub import snapshot_download
    return snapshot_download(
        source, repo_type="dataset", cache_dir=cache_dir,
        allow_patterns=["*.parquet", "**/*.parquet", "state_embeddings.pt",
                        "action_embeddings.pt", "metadata.json", "*/state_embeddings.pt",
                        "*/action_embeddings.pt", "*/metadata.json"])


def _native_embedding_dir(root: str, split: str | None) -> str | None:
    if os.path.isfile(root):
        return None
    candidates = []
    if split:
        candidates.append(os.path.join(root, split))
    candidates.append(root)
    candidates.extend(os.path.dirname(path) for path in glob.glob(
        os.path.join(root, "**", "metadata.json"), recursive=True))
    found = list(dict.fromkeys(path for path in candidates if is_embedding_dir(path)))
    if not found:
        return None
    if split and os.path.join(root, split) in found:
        return os.path.join(root, split)
    if len(found) != 1:
        raise ValueError(f"{root}: multiple embedding splits found; pass --split")
    return found[0]


def _parquet_files(root: str) -> list[str]:
    if os.path.isfile(root):
        return [root]
    files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no native embeddings or parquet files found in {root!r}")
    return files


def _fixed_list_to_numpy(col, dtype) -> np.ndarray:
    arr = col.combine_chunks()
    return arr.flatten().to_numpy(zero_copy_only=False).reshape(-1, arr.type.list_size).astype(dtype, copy=False)


def download(source: str, out_dir: str, cache_dir: str | None = None, dtype: str = "float16",
             force: bool = False, split: str | None = None) -> str:
    """Write a parquet embedding dataset (HF id, local dir or file) as an embedding dir.
    Native embedding repositories are used directly. An existing converted dir is
    reused unless ``force``."""
    if is_embedding_dir(out_dir) and not force:
        print(f"[hf-emb] reusing {out_dir}", flush=True)
        return out_dir
    root = _source_root(source, cache_dir)
    native = _native_embedding_dir(root, split)
    if native:
        print(f"[hf-emb] using native embeddings {native}", flush=True)
        return native
    import pyarrow.parquet as pq
    import torch
    files = _parquet_files(root)
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    schema = pq.ParquetFile(files[0]).schema_arrow
    missing = [c for c in EMB_COLS if c not in schema.names]
    if missing:
        raise ValueError(f"{source}: missing embedding columns {missing}")
    meta_cols = [k for k in META_KEYS if k in schema.names]
    dim = schema.field("state_embedding").type.list_size
    np_dtype = np.float16 if dtype == "float16" else np.float32
    n = sum(counts)
    state, action = np.empty((n, dim), np_dtype), np.empty((n, dim), np_dtype)
    samples: list[dict] = []
    at = 0
    for f, k in zip(files, counts):
        t = pq.read_table(f, columns=[*EMB_COLS, *meta_cols])
        state[at:at + k] = _fixed_list_to_numpy(t.column("state_embedding"), np_dtype)
        action[at:at + k] = _fixed_list_to_numpy(t.column("action_embedding"), np_dtype)
        cols = {c: t.column(c).to_pylist() for c in meta_cols}
        samples.extend({key: (cols[key][i] if key in cols else None) for key in META_KEYS} for i in range(k))
        at += k
        print(f"[hf-emb] {os.path.basename(f)}: {k} rows ({at}/{n})", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.from_numpy(state), os.path.join(out_dir, "state_embeddings.pt"))
    torch.save(torch.from_numpy(action), os.path.join(out_dir, "action_embeddings.pt"))
    tmp = os.path.join(out_dir, "metadata.json.tmp")
    with open(tmp, "w") as fo:
        json.dump({"num_samples": n, "hidden_size": dim, "source": source, "samples": samples}, fo)
    os.replace(tmp, os.path.join(out_dir, "metadata.json"))
    print(f"[hf-emb] wrote {out_dir}: {n} x {dim} ({dtype})", flush=True)
    return out_dir


def export(emb_dir: str, out_dir: str, shards: int = 8, dtype: str = "float16") -> list[str]:
    """Write an embedding dir as ``out_dir/data/train-XXXXX-of-XXXXX.parquet`` shards."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    samples = json.load(open(os.path.join(emb_dir, "metadata.json")))["samples"]
    se = torch.load(os.path.join(emb_dir, "state_embeddings.pt"), map_location="cpu", weights_only=True, mmap=True)
    ae = torch.load(os.path.join(emb_dir, "action_embeddings.pt"), map_location="cpu", weights_only=True, mmap=True)
    n, dim = se.shape
    if not (len(ae) == len(samples) == n):
        raise ValueError(f"{emb_dir}: {n} states, {len(ae)} actions, {len(samples)} samples")
    tdt = torch.float16 if dtype == "float16" else torch.float32
    per = math.ceil(n / shards)
    nsh = math.ceil(n / per)
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    written = []
    for i in range(nsh):
        a, b = i * per, min(n, (i + 1) * per)
        cols = {k: pa.array([s.get(k) for s in samples[a:b]]) for k in META_KEYS}
        for name, t in (("state_embedding", se), ("action_embedding", ae)):
            v = t[a:b].to(tdt).numpy().reshape(-1)
            cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(v), dim)
        path = os.path.join(data_dir, f"train-{i:05d}-of-{nsh:05d}.parquet")
        pq.write_table(pa.table(cols), path, compression="zstd")
        written.append(path)
        print(f"[hf-emb] {os.path.basename(path)}: rows {a}-{b}", flush=True)
    return written


def push(out_dir: str, repo_id: str, private: bool = False) -> None:
    """Upload an exported dir (``data/`` + optional ``README.md``) as a HF dataset."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="dataset")
    print(f"[hf-emb] pushed {out_dir} -> https://huggingface.co/datasets/{repo_id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="parquet dataset -> embedding dir")
    d.add_argument("source", nargs="?", default=DEEPSWE_TRAIN_DATASET,
                   help=f"HF dataset id, local dir, or parquet file (default: {DEEPSWE_TRAIN_DATASET})")
    d.add_argument("--out", required=True)
    d.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    d.add_argument("--cache-dir", default=None)
    d.add_argument("--split", default=None, help="native embedding subdirectory, e.g. train")
    d.add_argument("--force", action="store_true")
    e = sub.add_parser("export", help="embedding dir -> parquet shards")
    e.add_argument("emb_dir")
    e.add_argument("--out", required=True)
    e.add_argument("--shards", type=int, default=8)
    e.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    e.add_argument("--push", default=None, metavar="REPO_ID", help="also upload to this HF dataset")
    e.add_argument("--private", action="store_true")
    args = ap.parse_args()
    if args.cmd == "download":
        download(args.source, args.out, args.cache_dir, args.dtype, args.force, args.split)
    else:
        export(args.emb_dir, args.out, args.shards, args.dtype)
        if args.push:
            push(args.out, args.push, args.private)


if __name__ == "__main__":
    main()
