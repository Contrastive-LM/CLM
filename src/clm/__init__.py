"""CLM — contrastive language model inference engine and System One API.

Client (no torch needed):

    from clm import CLMClient, Choice, Noul, Score
    r = CLMClient().system_one(state, {"ok": Noul(instructions="Is this fine?")})

Engine (in-process, needs ``pip install -r requirements.txt`` from the repo and an embedder endpoint):

    from clm import Engine
    Engine().answer(state, {"ok": {"type": "noul", "instructions": "Is this fine?"}})

Fine-tuning (the same install; the encoder stays frozen, the two heads train on your data):

    import clm
    result = clm.finetune("data/train.jsonl")          # or a list of dicts
    result.engine().rank(state, candidates)

Server: ``clm-serve``.  Checkpoint: ``clm-download``.  Trainer: ``clm-finetune``.
"""
from .client import (Answer, Choice, ChoiceAnswer, CLMClient, CLMError, Noul, NoulAnswer, Question, Score,
                     ScoreAnswer, SystemOneResponse, Usage)
from .schema import answer_from_logits, answer_from_probs, build_pairs, candidates, state_text

__version__ = "0.1.0"
__all__ = ["CLMClient", "CLMError", "Noul", "Choice", "Score", "Question", "Answer", "NoulAnswer", "ChoiceAnswer",
           "ScoreAnswer", "SystemOneResponse", "Usage", "Engine", "Embedder", "HeadPair",
           "finetune", "FinetuneResult", "build_pairs", "candidates", "state_text", "answer_from_logits", "answer_from_probs", "__version__"]


def __getattr__(name):  # lazy: Engine / Embedder / HeadPair / finetune pull in numpy+torch only when used
    if name in ("finetune", "FinetuneResult"):
        from . import finetuning
        return getattr(finetuning, name)
    if name == "Engine":
        from .engine import Engine
        return Engine
    if name == "Embedder":
        from .embedder import Embedder
        return Embedder
    if name == "HeadPair":
        from .heads import HeadPair
        return HeadPair
    raise AttributeError(name)
