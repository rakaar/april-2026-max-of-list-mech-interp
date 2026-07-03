#!/usr/bin/env python3
"""Compare per-head value vectors against output readout directions in 16d space."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_value_vs_readout_pca.png"


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


def pca_rows(x: torch.Tensor, k: int = 2):
    centered = x - x.mean(dim=0, keepdim=True)
    u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    scores = u[:, :k] * s[:k]
    energy = s.square() / s.square().sum()
    return scores, energy, centered.mean(dim=0), vh[:k]


def pca_project(x: torch.Tensor, mean: torch.Tensor, components: torch.Tensor):
    return (x - mean) @ components.T


def draw_labeled_points(ax, scores, marker, label_prefix, color_values, title, alpha=1.0):
    xs = scores[:, 0].numpy()
    ys = scores[:, 1].numpy()
    ax.plot(xs, ys, color="#cbd5e1", linewidth=1.0, alpha=0.7)
    ax.scatter(
        xs,
        ys,
        c=color_values,
        cmap="viridis",
        s=58 if marker != "x" else 64,
        marker=marker,
        linewidths=1.7 if marker == "x" else 0.8,
        edgecolors="#111827" if marker != "x" else None,
        alpha=alpha,
        zorder=3,
    )
    for token, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(
            f"{label_prefix}{token}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0, color="#d1d5db", linewidth=0.8)
    ax.axvline(0, color="#d1d5db", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def main() -> None:
    model = load_model()
    layer = model.layers[0]
    e_numbers = model.tok_embed.weight.detach()[:10]  # 10 x 64
    u_numbers = model.unembed.weight.detach()[:10]  # 10 x 64
    w_o = layer.W_O.weight.detach()
    token_values = torch.arange(10)

    fig, axes = plt.subplots(3, 4, figsize=(17, 12))
    print("head,set,pc1,pc2,pc1_plus_pc2")

    for head_idx, head in enumerate(layer.heads):
        w_v = head.W_V.weight.detach()  # 16 x 64
        d_head = w_v.shape[0]
        w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]  # 64 x 16

        values = (w_v @ e_numbers.T).T  # 10 x 16
        readouts = (w_o_head.T @ u_numbers.T).T  # 10 x 16

        read_scores, read_energy, _, _ = pca_rows(readouts)
        value_scores, value_energy, _, _ = pca_rows(values)

        print(
            f"{head_idx},readout_WO_T_U,"
            f"{float(read_energy[0]):.6f},{float(read_energy[1]):.6f},"
            f"{float(read_energy[:2].sum()):.6f}"
        )
        print(
            f"{head_idx},value_WV_E,"
            f"{float(value_energy[0]):.6f},{float(value_energy[1]):.6f},"
            f"{float(value_energy[:2].sum()):.6f}"
        )

        draw_labeled_points(
            axes[0, head_idx],
            read_scores,
            marker="x",
            label_prefix="",
            color_values=token_values,
            title=(
                f"H{head_idx} readout W_O^T U\n"
                f"PC1 {100 * float(read_energy[0]):.1f}%, "
                f"PC1+2 {100 * float(read_energy[:2].sum()):.1f}%"
            ),
        )
        draw_labeled_points(
            axes[1, head_idx],
            value_scores,
            marker=".",
            label_prefix="",
            color_values=token_values,
            title=(
                f"H{head_idx} value W_V E\n"
                f"PC1 {100 * float(value_energy[0]):.1f}%, "
                f"PC1+2 {100 * float(value_energy[:2].sum()):.1f}%"
            ),
        )

        combined = torch.cat([readouts, values], dim=0)
        _, combined_energy, _, components = pca_rows(combined)
        combined_mean = combined.mean(dim=0)
        read_common = pca_project(readouts, combined_mean, components)
        value_common = pca_project(values, combined_mean, components)
        print(
            f"{head_idx},combined_basis,"
            f"{float(combined_energy[0]):.6f},{float(combined_energy[1]):.6f},"
            f"{float(combined_energy[:2].sum()):.6f}"
        )

        ax = axes[2, head_idx]
        draw_labeled_points(
            ax,
            read_common,
            marker="x",
            label_prefix="r",
            color_values=token_values,
            title=(
                f"H{head_idx} shared PCA overlay\n"
                f"PC1 {100 * float(combined_energy[0]):.1f}%, "
                f"PC1+2 {100 * float(combined_energy[:2].sum()):.1f}%"
            ),
            alpha=0.9,
        )
        xs = value_common[:, 0].numpy()
        ys = value_common[:, 1].numpy()
        ax.plot(xs, ys, color="#94a3b8", linewidth=1.0, alpha=0.65)
        ax.scatter(
            xs,
            ys,
            c=token_values,
            cmap="viridis",
            s=70,
            marker=".",
            alpha=0.95,
            zorder=4,
        )
        for token, (x, y) in enumerate(zip(xs, ys)):
            ax.annotate(f"v{token}", (x, y), xytext=(5, -10), textcoords="offset points", fontsize=8)
        ax.text(
            0.02,
            0.98,
            "x = readout, . = value",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#e5e7eb", "alpha": 0.85},
        )

    fig.suptitle("Model 1 per-head readout curves vs value-vector lines", y=0.995)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(f"wrote,{OUT}")


if __name__ == "__main__":
    main()

