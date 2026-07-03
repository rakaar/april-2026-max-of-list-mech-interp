#!/usr/bin/env python3
"""PCA of Model 1 number embeddings after each head's W_V map."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_wv_embedding_pca.png"


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
    centered = x - x.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    scores = u[:, :k] * s[:k]
    energy = s.square() / s.square().sum()
    return scores, energy


def main() -> None:
    model = load_model()
    number_embeddings = model.tok_embed.weight.detach()[:10]  # 10 x 64
    token_values = torch.arange(10, dtype=torch.float32)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    print("head,pc1,pc2,pc3,pc1_token_corr,pc1_order_asc,norm_order_desc")

    for head_idx, head in enumerate(model.layers[0].heads):
        w_v = head.W_V.weight.detach()  # 16 x 64

        # User's requested object is W_V @ W_E_numbers.T = 16 x 10.
        # Transpose to 10 x 16 so rows are number-token observations for PCA.
        value_vectors = (w_v @ number_embeddings.T).T
        scores, energy = pca_rows(value_vectors)
        norms = value_vectors.norm(dim=1)
        corr = torch.corrcoef(torch.stack([token_values, scores[:, 0]]))[0, 1]
        pc1_order = "-".join(str(int(i)) for i in torch.argsort(scores[:, 0]))
        norm_order = "-".join(str(int(i)) for i in torch.argsort(norms, descending=True))
        print(
            f"{head_idx},{float(energy[0]):.6f},{float(energy[1]):.6f},"
            f"{float(energy[2]):.6f},{float(corr):+.6f},{pc1_order},{norm_order}"
        )

        ax = axes[head_idx]
        xs = scores[:, 0].numpy()
        ys = scores[:, 1].numpy()
        ax.plot(xs, ys, color="#9ca3af", linewidth=1.2, alpha=0.75)
        ax.scatter(xs, ys, c=range(10), cmap="viridis", s=58, zorder=3)
        for token, (x, y) in enumerate(zip(xs, ys)):
            ax.annotate(str(token), (x, y), xytext=(5, 5), textcoords="offset points")
        ax.axhline(0, color="#d1d5db", linewidth=0.8)
        ax.axvline(0, color="#d1d5db", linewidth=0.8)
        ax.set_title(
            f"H{head_idx}: PC1 {100 * float(energy[0]):.1f}%, "
            f"PC1+2 {100 * float(energy[:2].sum()):.1f}%"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.suptitle("Model 1 W_V @ number-token embeddings", y=1.03)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"wrote,{OUT}")


if __name__ == "__main__":
    main()

