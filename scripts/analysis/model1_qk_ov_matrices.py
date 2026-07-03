#!/usr/bin/env python3
"""Plot token-token QK and OV circuit matrices for Model 1 attention heads."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "docs" / "assets"
QK_OUT = ASSET_DIR / "model1_qk_matrices.png"
OV_OUT = ASSET_DIR / "model1_ov_matrices.png"
COMBINED_OUT = ASSET_DIR / "model1_qk_ov_matrices.png"

TOKEN_LABELS = [str(i) for i in range(10)] + ["BOS", "SEP", "ANS", "EOS"]


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


def circuit_matrices(model):
    """Return per-head token-token QK and OV matrices.

    Row-vector implementation equations:
      QK[query, key] = E @ W_Q.T @ W_K @ E.T / sqrt(d_head)
      OV[source, out] = E @ W_V.T @ W_O_h.T @ U.T

    For plotting OV in transformer-circuits convention we transpose it to:
      OV[output_logit, source] = U @ W_O_h @ W_V @ E.T
    """
    embeddings = model.tok_embed.weight.detach()  # vocab x d_model
    unembed = model.unembed.weight.detach()  # vocab x d_model
    layer = model.layers[0]
    w_o = layer.W_O.weight.detach()

    qk_mats = []
    ov_mats = []
    for head_idx, head in enumerate(layer.heads):
        w_q = head.W_Q.weight.detach()
        w_k = head.W_K.weight.detach()
        w_v = head.W_V.weight.detach()
        d_head = w_v.shape[0]
        w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]

        qk = embeddings @ w_q.T @ w_k @ embeddings.T / (d_head**0.5)
        ov = unembed @ w_o_head @ w_v @ embeddings.T
        qk_mats.append(qk)
        ov_mats.append(ov)

    return qk_mats, ov_mats


def heatmap(ax, matrix: torch.Tensor, title: str, x_label: str, y_label: str):
    values = matrix.numpy()
    vmax = max(abs(values.min()), abs(values.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(values, cmap="coolwarm", norm=norm)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(TOKEN_LABELS)))
    ax.set_yticks(range(len(TOKEN_LABELS)))
    ax.set_xticklabels(TOKEN_LABELS, rotation=45, ha="left", fontsize=8)
    ax.set_yticklabels(TOKEN_LABELS, fontsize=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(x_label, fontsize=9, labelpad=10)
    ax.set_ylabel(y_label, fontsize=9)
    ax.tick_params(axis="both", length=0)
    return im


def plot_grid(mats, out_path: Path, title: str, x_label: str, y_label: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for head_idx, ax in enumerate(axes.flat):
        im = heatmap(ax, mats[head_idx], f"Head {head_idx}", x_label, y_label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)


def plot_combined(qk_mats, ov_mats) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for head_idx in range(4):
        im = heatmap(
            axes[0, head_idx],
            qk_mats[head_idx],
            f"H{head_idx} QK",
            "Key token",
            "Query token",
        )
        fig.colorbar(im, ax=axes[0, head_idx], fraction=0.046, pad=0.04)
        im = heatmap(
            axes[1, head_idx],
            ov_mats[head_idx],
            f"H{head_idx} OV",
            "Source token",
            "Output logit token",
        )
        fig.colorbar(im, ax=axes[1, head_idx], fraction=0.046, pad=0.04)
    fig.suptitle("Model 1 token-token QK and OV circuit matrices", y=0.995, fontsize=15)
    fig.tight_layout()
    fig.savefig(COMBINED_OUT, dpi=180)


def print_summary(qk_mats, ov_mats) -> None:
    print("matrix,head,row_axis,column_axis,min,max,absmax,row_argmax_for_number_rows")
    for name, mats, row_axis, col_axis in [
        ("QK", qk_mats, "query", "key"),
        ("OV", ov_mats, "output_logit", "source"),
    ]:
        for head_idx, mat in enumerate(mats):
            number_rows = mat[:10, :10]
            row_argmax = "-".join(str(int(i)) for i in number_rows.argmax(dim=1))
            print(
                f"{name},{head_idx},{row_axis},{col_axis},"
                f"{float(mat.min()):+.6f},{float(mat.max()):+.6f},"
                f"{float(mat.abs().max()):.6f},{row_argmax}"
            )


def main() -> None:
    model = load_model()
    qk_mats, ov_mats = circuit_matrices(model)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    plot_grid(qk_mats, QK_OUT, "Model 1 token-token QK matrices", "Key token", "Query token")
    plot_grid(
        ov_mats,
        OV_OUT,
        "Model 1 token-token OV matrices",
        "Source token",
        "Output logit token",
    )
    plot_combined(qk_mats, ov_mats)
    print_summary(qk_mats, ov_mats)
    print(f"wrote,{QK_OUT}")
    print(f"wrote,{OV_OUT}")
    print(f"wrote,{COMBINED_OUT}")


if __name__ == "__main__":
    main()

