"""HTTP contract for the optional Apple Silicon encoder."""

import base64
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(sys.platform == "darwin", "MLX requires Apple Silicon macOS")
class MlxEmbedServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        import numpy as np
        from fastapi.testclient import TestClient

        script = Path(__file__).resolve().parents[1] / "tools" / "mlx_embed_server.py"
        spec = importlib.util.spec_from_file_location("mlx_embed_server", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2]

        class Model:
            def model(self, tokens):
                return mx.array([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]])

        with patch.object(module, "load", return_value=(Model(), Tokenizer())):
            cls.client = TestClient(module.create_app("unused"))
        cls.np = np

    def test_float_and_base64_represent_the_same_normalized_vector(self):
        body = {"model": "qwen3-8b", "input": ["first", "second"]}
        floats = self.client.post("/v1/embeddings", json=body)
        encoded = self.client.post("/v1/embeddings", json={**body, "encoding_format": "base64"})
        self.assertEqual(floats.status_code, 200)
        self.assertEqual(encoded.status_code, 200)
        self.assertEqual(floats.json()["usage"]["prompt_tokens"], 4)
        for index, (left, right) in enumerate(zip(floats.json()["data"], encoded.json()["data"])):
            self.assertEqual(left["index"], index)
            self.assertEqual(right["index"], index)
            decoded = self.np.frombuffer(base64.b64decode(right["embedding"]), dtype="<f4")
            self.np.testing.assert_allclose(decoded, left["embedding"], atol=1e-6)
            self.np.testing.assert_allclose(decoded, [0.6, 0.8, 0.0], atol=1e-6)

    def test_rejects_unknown_model_and_empty_input(self):
        self.assertEqual(self.client.post("/v1/embeddings", json={"model": "other", "input": "x"}).status_code, 400)
        self.assertEqual(self.client.post("/v1/embeddings", json={"model": "qwen3-8b", "input": []}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
