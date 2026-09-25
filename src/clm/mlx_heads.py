"""MLX projection heads (Metal) matching the torch MLP in ``heads.make_head``."""
from __future__ import annotations

from typing import Any

import numpy as np

from .heads import HIDDEN, PROJ_DIM


def make_head_mlx(width: int, depth: int = 2, proj: int = PROJ_DIM, activation: str = "gelu",
                  layernorm: bool = False, residual: bool = False, hidden: int = HIDDEN):
    """``hidden -> width -> ... -> proj`` MLP on MLX."""
    import mlx.core as mx
    import mlx.nn as nn

    act_fn = {"gelu": nn.gelu, "relu": nn.relu, "silu": nn.silu}[activation]

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = [nn.Linear(width, width) for _ in range(max(0, depth - 2))]
            self.norms = [nn.LayerNorm(width) if layernorm else nn.Identity()
                          for _ in range(max(0, depth - 2))]
            self.out = nn.Linear(width, proj)
            self._residual = residual
            self._act = act_fn

        def __call__(self, x):
            x = self._act(self.inp(x))
            for lin, nrm in zip(self.hidden, self.norms):
                h = self._act(nrm(lin(x)))
                x = x + h if self._residual else h
            return self.out(x)

    return Head()


def load_state_dict_mlx(module, state: dict[str, Any]) -> None:
    """Copy a torch-style state dict (numpy-able tensors) into an mlx.nn.Module."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    flat = []
    for k, v in state.items():
        if hasattr(v, "detach"):
            arr = v.detach().cpu().numpy()
        else:
            arr = np.asarray(v)
        flat.append((k, mx.array(arr)))
    module.update(tree_unflatten(flat))


def n_params_mlx(module) -> int:
    from mlx.utils import tree_flatten
    return sum(int(p.size) for _, p in tree_flatten(module.parameters()))


def project_mlx(head, x: np.ndarray):
    """L2-normalised projections left as an mlx array on Metal."""
    import mlx.core as mx

    t = mx.array(np.asarray(x, dtype=np.float32))
    y = head(t)
    y = y / mx.maximum(mx.linalg.norm(y, ord=2, axis=-1, keepdims=True), 1e-12)
    mx.eval(y)
    return y
