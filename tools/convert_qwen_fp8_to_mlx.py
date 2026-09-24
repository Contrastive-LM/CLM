"""Convert Qwen's block-FP8 checkpoint to MLX MXFP8 without BF16 files.

Qwen's FP8 uses E4M3 weights with one inverse scale per 128x128 block.
MLX MXFP8 uses E4M3 weights with one E8M0 scale per 32 values. Conversion
briefly restores each linear layer in memory, then immediately requantizes it.
"""

import argparse
import gc
import json
import shutil
from pathlib import Path

import mlx.core as mx
import torch
from safetensors import safe_open


def restored_weight(weight: torch.Tensor, scale: torch.Tensor, block: list[int]) -> torch.Tensor:
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"expected 2D weight and scales, got {weight.shape}, {scale.shape}")
    rows, cols = weight.shape
    expected = ((rows + block[0] - 1) // block[0], (cols + block[1] - 1) // block[1])
    if tuple(scale.shape) != expected:
        raise ValueError(f"FP8 scale shape {tuple(scale.shape)} does not match {expected}")
    expanded = scale.float().repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
    return (weight.float() * expanded[:rows, :cols]).half()


def convert_tensor(name: str, get_tensor, block: list[int]) -> dict[str, mx.array]:
    tensor = get_tensor(name)
    scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
    if name.endswith(".weight") and get_tensor.has_key(scale_name):
        restored = restored_weight(tensor, get_tensor(scale_name), block)
        value = mx.array(restored.numpy())
    elif name.endswith(".weight") and tensor.ndim == 2:
        # Qwen leaves token embeddings and the output head uncompressed in its
        # FP8 release. Quantize those too so the Mac stores no full 2D weights.
        value = mx.array(tensor.float().numpy())
    else:
        if tensor.dtype == torch.bfloat16:
            value = mx.array(tensor.float().numpy()).astype(mx.float16)
        else:
            value = mx.array(tensor.numpy())
        mx.eval(value)
        return {name: value}
    if name.endswith(".weight") and tensor.ndim == 2:
        packed, scales = mx.quantize(value, group_size=32, bits=8, mode="mxfp8")
        mx.eval(packed, scales)
        return {name: packed, name.removesuffix(".weight") + ".scales": scales}
    raise AssertionError(f"unhandled tensor {name}")


def convert(source: Path, target: Path) -> None:
    config = json.loads((source / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [128, 128]:
        raise ValueError("source must be Qwen's 128x128 block-FP8 checkpoint")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    target.mkdir(parents=True, exist_ok=True)
    output_map = {}
    for filename in sorted(set(index["weight_map"].values())):
        output_file = target / filename
        if output_file.exists():
            with safe_open(output_file, framework="np") as reader:
                keys = list(reader.keys())
                for key in keys:
                    output_map[key] = filename
            print(f"reusing {filename}: {len(keys)} tensors", flush=True)
            continue
        arrays = {}
        with safe_open(source / filename, framework="pt", device="cpu") as reader:
            class Access:
                def __call__(self, key):
                    return reader.get_tensor(key)

                def has_key(self, key):
                    return key in reader.keys()

            access = Access()
            for key in reader.keys():
                if key.endswith(".weight_scale_inv"):
                    continue
                converted = convert_tensor(key, access, quant["weight_block_size"])
                arrays.update(converted)
                for new_key in converted:
                    output_map[new_key] = filename
        mx.save_safetensors(str(output_file), arrays, metadata={"format": "mlx"})
        print(f"saved {filename}: {len(arrays)} tensors", flush=True)
        del arrays
        gc.collect()
        mx.clear_cache()

    config.pop("quantization_config")
    config["torch_dtype"] = "float16"
    config["quantization"] = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "generation_config.json", "special_tokens_map.json"):
        src = source / name
        if src.exists():
            shutil.copy2(src, target / name)
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": output_map}, indent=2) + "\n"
    )
    (target / "conversion.json").write_text(json.dumps({
        "source": "Qwen/Qwen3-8B-FP8",
        "source_format": "block FP8 E4M3, 128x128",
        "output_format": "MLX MXFP8 E4M3, group 32",
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    convert(args.source, args.target)
