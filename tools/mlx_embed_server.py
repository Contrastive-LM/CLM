"""Serve Qwen3-8B's final-token hidden state for the CLM reference head.

This uses the causal model itself, unlike a separate embedding checkpoint.
"""

import argparse
import base64
from typing import Literal

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, HTTPException
from mlx_lm import load
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    truncate_prompt_tokens: int | None = None
    encoding_format: Literal['float', 'base64'] | None = None


def create_app(model_path: str) -> FastAPI:
    model, tokenizer = load(model_path)
    app = FastAPI()

    @app.get('/v1/models')
    def models():
        return {'object': 'list', 'data': [{'id': 'qwen3-8b', 'object': 'model'}]}

    @app.post('/v1/embeddings')
    def embeddings(request: EmbeddingRequest):
        if request.model != 'qwen3-8b':
            raise HTTPException(400, 'unknown model')
        texts = [request.input] if isinstance(request.input, str) else request.input
        if not texts or not all(isinstance(text, str) and text for text in texts):
            raise HTTPException(400, 'input must contain nonempty strings')
        vectors = []
        token_count = 0
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if request.truncate_prompt_tokens:
                tokens = tokens[-request.truncate_prompt_tokens:]
            token_count += len(tokens)
            states = model.model(mx.array(tokens, dtype=mx.int32)[None, :])
            last = states[0, -1].astype(mx.float32)
            mx.eval(last)
            vector = np.asarray(last)
            vector /= np.linalg.norm(vector) + 1e-12
            embedding = (base64.b64encode(vector.astype('<f4', copy=False).tobytes()).decode('ascii')
                         if request.encoding_format == 'base64' else vector.tolist())
            vectors.append({'object': 'embedding', 'index': len(vectors), 'embedding': embedding})
        return {'object': 'list', 'data': vectors, 'model': 'qwen3-8b',
                'usage': {'prompt_tokens': token_count, 'total_tokens': token_count}}

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=8091)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.model), host='127.0.0.1', port=args.port)
