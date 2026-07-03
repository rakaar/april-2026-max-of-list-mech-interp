#!/usr/bin/env python3
"""PCA of Model 1 number embeddings after each head's V and OV maps."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_head_value_ov_pca.png"


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


def pca_rows(x: torch.Tensor, k: int = 3):
    """Return PCA scores and explained variance ratios for row observations."""
    centered = x - x.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    scores = u[:, :k] * s[:k]
    variance = s.square()
    ratios = variance / variance.sum()
    return scores, ratios


def annotate_tokens(ax, scores: torch.Tensor, title: str) -> None:
    xs = scores[:, 0].numpy()
    ys = scores[:, 1].numpy()
    ax.plot(xs, ys, color="#9ca3af", linewidth=1, alpha=0.7)
    ax.scatter(xs, ys, c=range(10), cmap="viridis", s=48, zorder=3)
    for token, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(str(token), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#d1d5db", linewidth=0.8)
    ax.axvline(0, color="#d1d5db", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def main() -> None:
    model = load_model()
    layer = model.layers[0]
    embeddings = model.tok_embed.weight.detach()[:10]

    rows = []
    print("space,head,pc1,pc2,pc3,pc1_value_corr,pc1_monotonic")

    fig, axes = plt.subplots(4, 2, figsize=(10, 15))
    token_values = torch.arange(10, dtype=torch.float32)

    for head_idx, head in enumerate(layer.heads):
        w_v = head.W_V.weight.detach()
        d_head = w_v.shape[0]
        start = head_idx * d_head
        end = start + d_head
        w_o_head = layer.W_O.weight.detach()[:, start:end]

        value_space = embeddings @ w_v.T
        ov_space = value_space @ w_o_head.T

        for col_idx, (space_name, x) in enumerate(
            [("V", value_space), ("OV", ov_space)]
        ):
            scores, ratios = pca_rows(x)
            pc1_corr = torch.corrcoef(torch.stack([token_values, scores[:, 0]]))[0, 1]
            pc1 = scores[:, 0]
            monotonic = bool(
                all(pc1[i] <= pc1[i + 1] for i in range(9))
                or all(pc1[i] >= pc1[i + 1] for i in range(9))
            )
            print(
                f"{space_name},{head_idx},"
                f"{float(ratios[0]):.6f},{float(ratios[1]):.6f},{float(ratios[2]):.6f},"
                f"{float(pc1_corr):.6f},{monotonic}"
            )
            rows.append((space_name, head_idx, scores, ratios))
            title = (
                f"H{head_idx} {space_name}: "
                f"PC1 {100 * float(ratios[0]):.1f}%, "
                f"PC1+2 {100 * float(ratios[:2].sum()):.1f}%"
            )
            annotate_tokens(axes[head_idx, col_idx], scores, title)

    fig.suptitle("Model 1 number embeddings through each head's V and OV maps", y=0.995)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(f"wrote,{OUT}")


if __name__ == "__main__":
    main()

