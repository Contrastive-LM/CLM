#!/usr/bin/env python3
"""Download, verify, evaluate, or retrain the published Harvey CLM head."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(directory):
    checksums = json.loads((directory / 'checksums.json').read_text())
    for name, expected in checksums.items():
        path = directory / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Missing or changed release file: {path}')
    print(f'Verified {len(checksums)} files in {directory}', flush=True)


def run(command, cwd=REPO):
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    subprocess.run([str(x) for x in command], cwd=cwd, env=env, check=True)


def evaluate(data, checkpoint, output, source=REPO):
    output.mkdir(parents=True, exist_ok=True)
    for split in ('val', 'test'):
        run([sys.executable, source / 'evaluation/bon_eval.py',
             '--embeddings-dir', data / 'embeddings' / split,
             '--checkpoint', checkpoint, '--n', '8', '--window', '12',
             '--output', output / f'{split}.json'])


def train(args):
    from huggingface_hub import hf_hub_download
    provenance = json.loads((args.model / 'provenance.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / 'source'
    if source.exists():
        raise ValueError(f'Use a new output directory: {source} already exists')
    run(['git', 'clone', '--local', REPO, source])
    run(['git', 'checkout', '--detach', provenance['upstream_commit']], cwd=source)
    run(['git', 'apply', HERE / 'finetune.patch'], cwd=source)
    if digest(source / 'train/finetune.py') != provenance['trainer_sha256']:
        raise ValueError('Patched trainer differs from the recorded experiment')
    initial = provenance['initial_checkpoint']
    checkpoint = Path(hf_hub_download(initial['repo'], initial['file']))
    if digest(checkpoint) != initial['sha256']:
        raise ValueError('Released starting checkpoint has changed')
    trained = args.output / 'trained'
    run([sys.executable, source / 'train/finetune.py', '--task', 'clm',
         '--emb-dir', args.data / 'embeddings/parthenon_cafl',
         '--init-ckpt', checkpoint, '--out-dir', trained,
         '--batch', '2048', '--epochs', '20', '--weight-decay', '0',
         '--val-frac', '0.1', '--patience', '5', '--seed', '1234',
         '--sampler', 'random', '--clm-select-metric', 'within_task_top1'], cwd=source)
    actual = json.loads((trained / 'task_split.json').read_text())
    expected = json.loads((args.data / 'task_split.json').read_text())
    if actual != expected:
        raise ValueError('Training task split differs from the recorded experiment')
    evaluate(args.data, trained / 'best_head.pt', args.output / 'evaluation', source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['download', 'verify', 'evaluate', 'train'])
    parser.add_argument('--data', type=Path, default=Path('artifacts/harvey/data'))
    parser.add_argument('--model', type=Path, default=Path('artifacts/harvey/model'))
    parser.add_argument('--output', type=Path, default=Path('runs/harvey-reproduction'))
    args = parser.parse_args()
    for name in ('data', 'model', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if args.action == 'download':
        from huggingface_hub import snapshot_download
        release = json.loads((HERE / 'release.json').read_text())
        for kind, path in [('dataset', args.data), ('model', args.model)]:
            item = release[kind]
            snapshot_download(item['repo'], repo_type=kind,
                              revision=item['revision'], local_dir=path)
    verify(args.data)
    verify(args.model)
    if args.action == 'evaluate':
        evaluate(args.data, args.model / 'head.pt', args.output)
        for split in ('val', 'test'):
            actual = json.loads((args.output / f'{split}.json').read_text())
            expected = json.loads((args.model / f'{split}_results.json').read_text())
            for key in ('n_tasks', 'n_rollouts', 'selectors'):
                if actual[key] != expected[key]:
                    raise ValueError(f'{split} result differs: {key}')
        print('Validation and test selections match the published results.', flush=True)
    elif args.action == 'train':
        train(args)


if __name__ == '__main__':
    main()
