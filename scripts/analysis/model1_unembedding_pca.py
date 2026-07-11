#!/usr/bin/env python3
"""Run PCA on Model 1 full-vocabulary and digit unembedding vectors."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_unembedding_pca.png"
OUT_3D = ROOT / "docs" / "assets" / "model1_unembedding_pca_3d.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_unembedding_pca.json"
TOKEN_LABELS = [str(i) for i in range(10)] + ["BOS", "SEP", "ANS", "EOS"]
THRESHOLDS = (0.90, 0.95, 0.99)


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


def pca(x: torch.Tensor) -> dict[str, torch.Tensor | int | dict[str, int]]:
    mean = x.mean(dim=0)
    centered = x - mean
    _, singular_values, directions = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    explained = energy / energy.sum()
    cumulative = torch.cumsum(explained, dim=0)
    scores = centered @ directions.T
    numerical_rank = int(torch.linalg.matrix_rank(centered))
    threshold_pcs = {
        f"{int(threshold * 100)}pct": int(torch.searchsorted(cumulative, threshold).item() + 1)
        for threshold in THRESHOLDS
    }
    return {
        "mean": mean,
        "centered": centered,
        "singular_values": singular_values,
        "directions": directions,
        "explained": explained,
        "cumulative": cumulative,
        "scores": scores,
        "numerical_rank": numerical_rank,
        "threshold_pcs": threshold_pcs,
    }


def serializable_summary(name: str, x: torch.Tensor, result: dict) -> dict:
    scores = result["scores"]
    explained = result["explained"]
    cumulative = result["cumulative"]
    singular_values = result["singular_values"]
    max_centered_rank = min(x.shape[0] - 1, x.shape[1])
    return {
        "name": name,
        "token_vector_shape": list(x.shape),
        "equivalent_column_matrix_shape": [x.shape[1], x.shape[0]],
        "centered_matrix_numerical_rank": result["numerical_rank"],
        "maximum_centered_rank": max_centered_rank,
        "singular_values": [float(value) for value in singular_values],
        "explained_variance_ratio": [float(value) for value in explained],
        "cumulative_explained_variance": [float(value) for value in cumulative],
        "pcs_for_variance_threshold": result["threshold_pcs"],
        "pc_scores": [[float(value) for value in row] for row in scores],
    }


def plot_scatter(ax, scores: torch.Tensor, labels: list[str], title: str, full_vocab: bool) -> None:
    scores_np = scores[:, :2].numpy()
    if full_vocab:
        ax.scatter(
            scores_np[:10, 0],
            scores_np[:10, 1],
            c=np.arange(10),
            cmap="viridis",
            s=62,
            edgecolor="white",
            linewidth=0.7,
            label="digits",
        )
        ax.scatter(
            scores_np[10:, 0],
            scores_np[10:, 1],
            color="#dc2626",
            marker="s",
            s=68,
            edgecolor="white",
            linewidth=0.7,
            label="special tokens",
        )
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.scatter(
            scores_np[:, 0],
            scores_np[:, 1],
            c=np.arange(len(labels)),
            cmap="viridis",
            s=72,
            edgecolor="white",
            linewidth=0.7,
        )

    full_offsets = {
        "0": (6, 4),
        "1": (6, -10),
        "2": (6, -8),
        "3": (6, 4),
        "4": (6, 4),
        "5": (6, -10),
        "6": (6, 4),
        "7": (6, -10),
        "8": (6, 12),
        "9": (6, 4),
        "BOS": (8, 15),
        "SEP": (8, 1),
        "ANS": (8, -13),
        "EOS": (8, 4),
    }
    digit_offsets = {
        "0": (6, 4),
        "1": (6, 4),
        "2": (6, -9),
        "3": (6, -9),
        "4": (6, -9),
        "5": (6, -9),
        "6": (6, 4),
        "7": (6, -9),
        "8": (6, 4),
        "9": (6, 4),
    }
    offsets = full_offsets if full_vocab else digit_offsets
    for idx, label in enumerate(labels):
        ax.annotate(
            label,
            xy=(scores_np[idx, 0], scores_np[idx, 1]),
            xytext=offsets[label],
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0.0, color="#9ca3af", linewidth=0.7)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.grid(alpha=0.18)


def plot_scree(ax, result: dict, title: str) -> None:
    explained = result["explained"].numpy()
    cumulative = result["cumulative"].numpy()
    xs = np.arange(1, len(explained) + 1)
    ax.bar(xs, explained, color="#2563eb", alpha=0.78, label="individual")
    ax.plot(xs, cumulative, color="#dc2626", marker="o", markersize=3.5, label="cumulative")
    for threshold in THRESHOLDS:
        ax.axhline(threshold, color="#6b7280", linewidth=0.7, linestyle="--", alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Fraction of centered variance")
    ax.set_xticks(xs)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)


def plot_3d(ax, result: dict, labels: list[str], title: str, full_vocab: bool) -> None:
    scores = result["scores"][:, :3].numpy()
    explained = result["explained"]

    # Connect the digit tokens in numeric order to reveal the learned trajectory.
    ax.plot(
        scores[:10, 0],
        scores[:10, 1],
        scores[:10, 2],
        color="#6b7280",
        linewidth=1.1,
        alpha=0.7,
        label="digit order 0 to 9",
    )
    ax.scatter(
        scores[:10, 0],
        scores[:10, 1],
        scores[:10, 2],
        c=np.arange(10),
        cmap="viridis",
        s=62,
        depthshade=False,
        edgecolor="white",
        linewidth=0.7,
        label="digits" if full_vocab else None,
    )
    if full_vocab:
        ax.scatter(
            scores[10:, 0],
            scores[10:, 1],
            scores[10:, 2],
            color="#dc2626",
            marker="s",
            s=68,
            depthshade=False,
            edgecolor="white",
            linewidth=0.7,
            label="special tokens",
        )

    spans = np.maximum(np.ptp(scores, axis=0), 1e-6)
    special_z_offsets = {"BOS": 0.10, "SEP": 0.0, "ANS": -0.10, "EOS": 0.03}
    for idx, label in enumerate(labels):
        x_offset = 0.025 * spans[0]
        y_offset = 0.0
        z_offset = 0.025 * spans[2]
        if full_vocab and label in special_z_offsets:
            z_offset = special_z_offsets[label] * spans[2]
            if label == "SEP":
                y_offset = 0.018 * spans[1]
            elif label == "ANS":
                y_offset = -0.018 * spans[1]
        ax.text(
            scores[idx, 0] + x_offset,
            scores[idx, 1] + y_offset,
            scores[idx, 2] + z_offset,
            label,
            fontsize=8,
        )

    ax.set_title(title, pad=14)
    ax.set_xlabel(f"PC1 ({float(explained[0]):.1%})", labelpad=8)
    ax.set_ylabel(f"PC2 ({float(explained[1]):.1%})", labelpad=8)
    ax.set_zlabel(f"PC3 ({float(explained[2]):.1%})", labelpad=8)
    ax.view_init(elev=22, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.15, 1.0, 0.85))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=False, fontsize=8)


def main() -> None:
    torch.manual_seed(0)
    model, config = load_model()
    full_unembedding = model.unembed.weight.detach().float()  # 14 token vectors x 64 dimensions
    digit_unembedding = full_unembedding[:10]

    full_result = pca(full_unembedding)
    digit_result = pca(digit_unembedding)

    digit_scores = digit_result["scores"]
    digit_values = torch.arange(10, dtype=digit_scores.dtype)
    digit_pc_correlations = [
        float(torch.corrcoef(torch.stack([digit_values, digit_scores[:, pc_idx]]))[0, 1])
        for pc_idx in range(min(3, digit_scores.shape[1]))
    ]

    summary = {
        "description": (
            "PCA of Model 1 unembedding token vectors. PyTorch stores W_U as "
            "vocab_size x d_model (14 x 64); this is equivalent to the user's "
            "64 x vocab_size column-vector convention. Tokens are observations "
            "and residual-stream dimensions are PCA features. Each set is centered "
            "across its own tokens before SVD."
        ),
        "hf_repo": "andyrdt/04_2026_puzzle_1a",
        "model_config": config,
        "token_labels": TOKEN_LABELS,
        "full_vocabulary": serializable_summary("all 14 tokens", full_unembedding, full_result),
        "digits_only": serializable_summary("digits 0-9", digit_unembedding, digit_result),
        "digits_only_pc_correlation_with_digit_value": digit_pc_correlations,
        "pca_sign_note": "Each PCA direction has arbitrary sign; correlation signs may flip without changing geometry.",
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), constrained_layout=True)
    plot_scatter(
        axes[0, 0],
        full_result["scores"],
        TOKEN_LABELS,
        "All-vocabulary unembedding PCA: 14 tokens in 64D",
        full_vocab=True,
    )
    plot_scatter(
        axes[0, 1],
        digit_result["scores"],
        TOKEN_LABELS[:10],
        "Digit-only unembedding PCA: 10 digits in 64D",
        full_vocab=False,
    )
    plot_scree(axes[1, 0], full_result, "All vocabulary: variance spectrum")
    plot_scree(axes[1, 1], digit_result, "Digits only: variance spectrum")
    fig.suptitle("Model 1: PCA of the unembedding matrix", fontsize=15)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)

    fig_3d = plt.figure(figsize=(15.0, 6.8), constrained_layout=True)
    full_ax = fig_3d.add_subplot(1, 2, 1, projection="3d")
    digit_ax = fig_3d.add_subplot(1, 2, 2, projection="3d")
    plot_3d(
        full_ax,
        full_result,
        TOKEN_LABELS,
        "All-vocabulary unembedding: top 3 PCs",
        full_vocab=True,
    )
    plot_3d(
        digit_ax,
        digit_result,
        TOKEN_LABELS[:10],
        "Digit-only unembedding: top 3 PCs",
        full_vocab=False,
    )
    fig_3d.suptitle("Model 1: 3D PCA of the unembedding matrix", fontsize=15)
    fig_3d.savefig(OUT_3D, dpi=180, facecolor="white")
    plt.close(fig_3d)

    print("set,pc1,pc2,pc3,pc1_plus_pc2,pc1_plus_pc2_plus_pc3,rank,pcs90,pcs95,pcs99")
    for name, result in (("full_vocab", full_result), ("digits_only", digit_result)):
        explained = result["explained"]
        cumulative = result["cumulative"]
        thresholds = result["threshold_pcs"]
        print(
            f"{name},{float(explained[0]):.6f},{float(explained[1]):.6f},"
            f"{float(explained[2]):.6f},{float(cumulative[1]):.6f},"
            f"{float(cumulative[2]):.6f},{result['numerical_rank']},"
            f"{thresholds['90pct']},{thresholds['95pct']},{thresholds['99pct']}"
        )
    print("digit_pc_correlations," + ",".join(f"{value:+.6f}" for value in digit_pc_correlations))
    print(f"wrote,{OUT}")
    print(f"wrote,{OUT_3D}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
