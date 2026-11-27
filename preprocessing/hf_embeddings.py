#!/usr/bin/env python3
"""Convert between embedding dirs and Hugging Face parquet datasets.

The converter lives in the package (``src/clm/hf_embeddings.py``); this script
runs it from a clone::

    python preprocessing/hf_embeddings.py download Contrastive-LM/deepswe-clm-train-embeddings-8k \\
        --out data/deepswe_train
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from clm.hf_embeddings import main  # noqa: E402

if __name__ == "__main__":
    main()
