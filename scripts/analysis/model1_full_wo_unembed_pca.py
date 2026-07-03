#!/usr/bin/env python3
"""PCA of full W_O-pulled number unembedding directions for Model 1."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_full_wo_unembed_pca.png"


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
    u_numbers = model.unembed.weight.detach()[:10]
    w_o = model.layers[0].W_O.weight.detach()

    # Rows are the 10 number-output directions pulled back through full W_O
    # into concatenated-head output space.
    observations = (w_o.T @ u_numbers.T).T
    centered = observations - observations.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    scores = u[:, :3] * s[:3]
    energy = s.square() / s.square().sum()

    token_values = torch.arange(10, dtype=torch.float32)
    print("pc,energy,corr_with_token,monotonic,coords")
    for pc_idx in range(3):
        coord = scores[:, pc_idx]
        corr = torch.corrcoef(torch.stack([token_values, coord]))[0, 1]
        monotonic = bool(
            all(coord[i] <= coord[i + 1] for i in range(9))
            or all(coord[i] >= coord[i + 1] for i in range(9))
        )
        coords = " ".join(f"{token}:{float(coord[token]):+.6f}" for token in range(10))
        print(
            f"PC{pc_idx + 1},{float(energy[pc_idx]):.6f},"
            f"{float(corr):+.6f},{monotonic},{coords}"
        )

    norms = observations.norm(dim=1)
    norm_corr = torch.corrcoef(torch.stack([token_values, norms]))[0, 1]
    norm_monotonic = bool(
        all(norms[i] <= norms[i + 1] for i in range(9))
        or all(norms[i] >= norms[i + 1] for i in range(9))
    )
    print(
        f"norms,corr_with_token={float(norm_corr):+.6f},"
        f"monotonic={norm_monotonic},"
        + " ".join(f"{token}:{float(norms[token]):.6f}" for token in range(10))
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = scores[:, 0].numpy()
    ys = scores[:, 1].numpy()
    ax.plot(xs, ys, color="#9ca3af", linewidth=1.25, alpha=0.8)
    ax.scatter(xs, ys, c=range(10), cmap="viridis", s=64, zorder=3)
    for token, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(str(token), (x, y), xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, color="#d1d5db", linewidth=0.8)
    ax.axvline(0, color="#d1d5db", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({100 * float(energy[0]):.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * float(energy[1]):.1f}% var)")
    ax.set_title("Full W_O-pulled number unembedding directions")
    fig.tight_layout()
    fig.savefig(OUT, dpi=180)
    print(f"wrote,{OUT}")


if __name__ == "__main__":
    main()

