"""Serve Qwen3 last-token embeddings on Apple Silicon through the OpenAI API."""
from __future__ import annotations

import argparse
import asyncio
import base64
import threading

import numpy as np
from fastapi import FastAPI, HTTPException, Request

MODEL = "Qwen/Qwen3-8B"
SERVED_MODEL = "qwen3-8b"


class MpsEncoder:
    def __init__(self, model: str = MODEL, device: str = "mps", max_tokens: int = 2048,
                 batch_size: int = 1, dtype: str = "auto"):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS is unavailable; check Apple Silicon and the torch installation")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.device, self.max_tokens, self.batch_size = device, max_tokens, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"
        if dtype not in ("auto", "float16", "bfloat16"):
            raise ValueError("dtype must be auto, float16 or bfloat16")
        if device == "mps":
            checkpoint_dtype = getattr(AutoConfig.from_pretrained(model), "dtype", None)
            requested = ("bfloat16" if checkpoint_dtype == torch.bfloat16 else "float16") if dtype == "auto" else dtype
            load_dtype = torch.bfloat16 if requested == "bfloat16" else torch.float16
            if load_dtype == torch.bfloat16:
                try:
                    probe = torch.ones((1, 1), dtype=load_dtype, device="mps")
                    (probe @ probe).cpu()  # force execution; older MPS devices cannot run BF16
                except (RuntimeError, TypeError) as exc:
                    if dtype != "auto":
                        raise RuntimeError("BF16 is unavailable on this MPS device") from exc
                    load_dtype = torch.float16
        else:
            load_dtype = torch.float32
        self.dtype = str(load_dtype).removeprefix("torch.")
        if device == "mps":
            # Transformers' allocator warmup requests one buffer larger than
            # Metal allows for Qwen3-8B; load its shards directly instead.
            from contextlib import nullcontext
            from unittest.mock import patch
            from transformers import modeling_utils

            original_warmup = getattr(modeling_utils, "caching_allocator_warmup", None)

            def warmup_unless_mps(model, expanded_device_map, hf_quantizer):
                if expanded_device_map and all(
                    str(value).startswith("mps") for value in expanded_device_map.values()
                ):
                    return
                return original_warmup(model, expanded_device_map, hf_quantizer)

            warmup_guard = (patch.object(modeling_utils, "caching_allocator_warmup", warmup_unless_mps)
                            if original_warmup is not None else nullcontext())
            with warmup_guard:
                self.model = AutoModel.from_pretrained(
                    model, dtype=load_dtype, device_map=device, low_cpu_mem_usage=True
                ).eval()
        else:
            self.model = AutoModel.from_pretrained(
                model, dtype=load_dtype, device_map=device, low_cpu_mem_usage=True
            ).eval()
        self._lock = threading.Lock()

    def embed(self, inputs: list[str] | list[list[int]], max_tokens: int | None = None) -> tuple[np.ndarray, int]:
        import torch

        limit = min(max_tokens or self.max_tokens, self.max_tokens)
        out, count = [], 0
        with self._lock, torch.inference_mode():
            for start in range(0, len(inputs), self.batch_size):
                chunk = inputs[start:start + self.batch_size]
                if isinstance(chunk[0], str):
                    batch = self.tokenizer([s or " " for s in chunk], padding=True, truncation=True,
                                           max_length=limit, return_tensors="pt")
                else:
                    rows = [ids[-limit:] for ids in chunk]
                    batch = self.tokenizer.pad({"input_ids": rows}, padding=True, return_tensors="pt")
                count += int(batch["attention_mask"].sum())
                indices = batch["attention_mask"].sum(dim=1) - 1
                batch = {k: v.to(self.device) for k, v in batch.items()}
                hidden = self.model(**batch, use_cache=False).last_hidden_state
                vectors = hidden[torch.arange(len(chunk), device=self.device), indices.to(self.device)]
                out.extend(vectors.float().cpu().numpy())
        return np.stack(out), count


def create_app(encoder: MpsEncoder, served_model: str = SERVED_MODEL) -> FastAPI:
    app = FastAPI(title="CLM MPS embedding server")

    @app.get("/health")
    def health():
        return {"ok": True, "model": served_model, "device": encoder.device, "dtype": encoder.dtype}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": served_model, "object": "model"}]}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(422, "body must be JSON") from exc
        if not isinstance(body, dict) or body.get("model") != served_model:
            raise HTTPException(422, f"model must be {served_model!r}")
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list) or not inputs or not all(
            isinstance(x, str) or (isinstance(x, list) and bool(x) and all(isinstance(t, int) and t >= 0 for t in x))
            for x in inputs
        ) or any(isinstance(x, str) != isinstance(inputs[0], str) for x in inputs):
            raise HTTPException(422, "input must be text or a non-empty list of texts or token ID lists")
        encoding = body.get("encoding_format", "float")
        if encoding not in ("float", "base64"):
            raise HTTPException(422, "encoding_format must be float or base64")
        limit = body.get("truncate_prompt_tokens")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise HTTPException(422, "truncate_prompt_tokens must be a positive integer")
        try:
            vectors, tokens = await asyncio.get_running_loop().run_in_executor(None, encoder.embed, inputs, limit)
        except (ValueError, IndexError) as exc:
            raise HTTPException(422, str(exc)) from exc
        data = []
        for i, vector in enumerate(vectors):
            arr = np.asarray(vector, dtype="<f4")
            value = base64.b64encode(arr.tobytes()).decode("ascii") if encoding == "base64" else arr.tolist()
            data.append({"object": "embedding", "index": i, "embedding": value})
        return {"object": "list", "data": data, "model": served_model,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL, help="local path or Hugging Face model ID")
    parser.add_argument("--served-model-name", default=SERVED_MODEL)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto",
                        help="MPS weight dtype; auto uses BF16 when the checkpoint and Mac support it")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="encoder microbatch size (default 1 to limit unified-memory use)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    if args.max_model_len < 1:
        parser.error("--max-model-len must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    encoder = MpsEncoder(args.model, args.device, args.max_model_len, args.batch_size, args.dtype)
    print(f"[clm] {args.model} embeddings on {args.device} ({encoder.dtype}), port {args.port}", flush=True)
    import uvicorn
    uvicorn.run(create_app(encoder, args.served_model_name), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
