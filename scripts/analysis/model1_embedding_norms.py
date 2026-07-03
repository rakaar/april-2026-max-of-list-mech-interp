#!/usr/bin/env python3
"""Inspect Model 1 token embedding norms."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_token_embedding_norms.png"


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
    return model


def main() -> None:
    model = load_model()
    embeddings = model.tok_embed.weight.detach()
    number_embeddings = embeddings[:10]
    norms = number_embeddings.norm(dim=1)

    token_values = torch.arange(10, dtype=torch.float32)
    pearson = torch.corrcoef(torch.stack([token_values, norms]))[0, 1]

    print("token,norm")
    for token, norm in enumerate(norms.tolist()):
        print(f"{token},{norm:.6f}")
    print(f"pearson_token_value_vs_norm,{float(pearson):.6f}")
    print(f"monotonic_increasing,{all(norms[i] <= norms[i + 1] for i in range(9))}")
    print(f"largest_norm_token,{int(norms.argmax())}")
    print(f"smallest_norm_token,{int(norms.argmin())}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#1f77b4"] * 10
    colors[int(norms.argmax())] = "#d62728"
    ax.bar(range(10), norms.numpy(), color=colors)
    ax.set_xticks(range(10))
    ax.set_xlabel("Number token")
    ax.set_ylabel("Embedding L2 norm")
    ax.set_title("Model 1 token embedding magnitudes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180)
    print(f"wrote,{OUT}")


if __name__ == "__main__":
    main()

