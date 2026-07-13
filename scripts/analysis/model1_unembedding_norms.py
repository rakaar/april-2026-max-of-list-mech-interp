#!/usr/bin/env python3
"""Inspect Model 1 digit unembedding row norms."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_unembedding_norms.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_unembedding_norms.json"


def load_model():
    model_py_path = hf_hub_download("andyrdt/04_2026_puzzle_1a", "model.py")
    spec = importlib.util.spec_from_file_location("model", model_py_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config_path = hf_hub_download("andyrdt/04_2026_puzzle_1a", "config.json")
    weights_path = hf_hub_download("andyrdt/04_2026_puzzle_1a", "model.pt")
    config = json.loads(Path(config_path).read_text())["model"]

    model = module.AttentionOnlyTransformer.from_config(config)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, config


def is_monotonic_increasing(values: torch.Tensor) -> bool:
    return bool(torch.all(values[:-1] <= values[1:]))


def is_monotonic_decreasing(values: torch.Tensor) -> bool:
    return bool(torch.all(values[:-1] >= values[1:]))


def main() -> None:
    model, config = load_model()
    digit_unembeddings = model.unembed.weight.detach()[:10]
    norms = digit_unembeddings.norm(dim=1)
    digits = torch.arange(10, dtype=torch.float32)
    pearson = torch.corrcoef(torch.stack([digits, norms]))[0, 1]
    order_desc = torch.argsort(norms, descending=True)

    result = {
        "description": (
            "Model 1 digit unembedding row L2 norms. The digit rows are "
            "model.unembed.weight[:10], shape 10 x d_model, and logits are "
            "residual @ W_U[d]."
        ),
        "hf_repo": "andyrdt/04_2026_puzzle_1a",
        "model_config": config,
        "tensor_shape": list(digit_unembeddings.shape),
        "norms": [float(x) for x in norms],
        "pearson_digit_value_vs_norm": float(pearson),
        "monotonic_increasing": is_monotonic_increasing(norms),
        "monotonic_decreasing": is_monotonic_decreasing(norms),
        "largest_norm_digit": int(norms.argmax()),
        "smallest_norm_digit": int(norms.argmin()),
        "norm_order_descending": [int(x) for x in order_desc],
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    print("digit,unembedding_l2_norm")
    for digit, norm in enumerate(norms.tolist()):
        print(f"{digit},{norm:.6f}")
    print(f"pearson_digit_value_vs_norm,{float(pearson):.6f}")
    print(f"monotonic_increasing,{result['monotonic_increasing']}")
    print(f"monotonic_decreasing,{result['monotonic_decreasing']}")
    print(f"largest_norm_digit,{result['largest_norm_digit']}")
    print(f"smallest_norm_digit,{result['smallest_norm_digit']}")

    colors = ["#2563eb"] * 10
    colors[int(norms.argmax())] = "#dc2626"
    colors[int(norms.argmin())] = "#059669"

    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    ax.bar(range(10), norms.numpy(), color=colors, alpha=0.88)
    ax.plot(range(10), norms.numpy(), color="#111827", linewidth=1.2, marker="o")
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Unembedding row L2 norm")
    ax.set_title("Model 1: digit unembedding vector norms")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0.0, float(norms.max()) * 1.16)

    for digit, norm in enumerate(norms.tolist()):
        ax.text(digit, norm + float(norms.max()) * 0.025, f"{norm:.2f}", ha="center", fontsize=8)

    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
