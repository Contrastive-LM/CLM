#!/usr/bin/env python3
"""Fine-tune the CLM projection heads: ``python train/finetune.py --task clm ...``.

The trainer lives in the package (``src/clm/trainer.py``) so that it ships with
``pip install contrastive-lm``; this script runs it from a clone. The same
entry point is installed as ``clm-finetune``, and ``clm.finetune(...)`` is the
Python API on top of it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from clm.trainer import main  # noqa: E402

if __name__ == "__main__":
    main()
