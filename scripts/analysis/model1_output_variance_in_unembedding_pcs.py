#!/usr/bin/env python3
"""Measure centered W_O row variance along the digit-unembedding PCs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
JSON_OUT = ROOT / "docs" / "assets" / "model1_output_variance_in_unembedding_pcs.json"
PNG_OUT = ROOT / "docs" / "assets" / "model1_output_variance_in_unembedding_pcs.png"
HF_REPO = "andyrdt/04_2026_puzzle_1a"
PC_COLORS = ["#2563eb", "#16a34a", "#dc2626"]


def load_model():
    model_py_path = hf_hub_download(HF_REPO, "model.py")
    spec = importlib.util.spec_from_file_location("model", model_py_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config_path = hf_hub_download(HF_REPO, "config.json")
    weights_path = hf_hub_download(HF_REPO, "model.pt")
    config = json.loads(Path(config_path).read_text())["model"]

    model = module.AttentionOnlyTransformer.from_config(config)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def variance_capture(matrix: torch.Tensor, basis: torch.Tensor) -> dict:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (centered.shape[0] - 1)
    total_variance = torch.trace(covariance)
    variance_by_pc = torch.diagonal(basis.T @ covariance @ basis)
    fraction_by_pc = variance_by_pc / total_variance

    projected = centered @ basis
    direct_variance_by_pc = projected.square().sum(dim=0) / (centered.shape[0] - 1)
    if not torch.allclose(variance_by_pc, direct_variance_by_pc, rtol=1e-10, atol=1e-12):
        raise AssertionError("covariance and direct projection calculations disagree")

    projected_reconstruction = projected @ basis.T
    reconstructed_fraction = (
        projected_reconstruction.square().sum() / centered.square().sum()
    )
    if not torch.allclose(
        fraction_by_pc.sum(), reconstructed_fraction, rtol=1e-10, atol=1e-12
    ):
        raise AssertionError("variance and reconstruction-energy fractions disagree")

    return {
        "shape": list(matrix.shape),
        "n_centered_rows": int(matrix.shape[0]),
        "total_centered_variance": float(total_variance),
        "variance_by_unembedding_pc": [float(value) for value in variance_by_pc],
        "fraction_by_unembedding_pc": [float(value) for value in fraction_by_pc],
        "top3_captured_variance": float(variance_by_pc.sum()),
        "top3_captured_fraction": float(fraction_by_pc.sum()),
        "top3_captured_percent": 100.0 * float(fraction_by_pc.sum()),
        "residual_fraction_outside_top3": 1.0 - float(fraction_by_pc.sum()),
    }


def plot_results(results: dict) -> None:
    labels = ["H0", "H1", "H2", "H3", "All heads"]
    fractions = np.asarray(
        [results["matrices"][label]["fraction_by_unembedding_pc"] for label in labels]
    )
    totals = fractions.sum(axis=1)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.5, 6.4), constrained_layout=True)
    bottom = np.zeros(len(labels))
    for pc_idx in range(3):
        values = 100.0 * fractions[:, pc_idx]
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=PC_COLORS[pc_idx],
            width=0.68,
            label=f"Unembedding PC{pc_idx + 1}",
        )
        for bar_idx, value in enumerate(values):
            if value >= 3.0:
                ax.text(
                    bar_idx,
                    bottom[bar_idx] + value / 2.0,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    weight="bold",
                )
        bottom += values

    for bar_idx, total in enumerate(100.0 * totals):
        ax.text(
            bar_idx,
            total + 1.4,
            f"{total:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
            color="#111827",
        )

    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 110.0)
    ax.set_ylabel("Fraction of centered output-row variance")
    ax.set_title("Model 1: output-matrix variance captured by digit-unembedding PCs")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper center", ncol=3, frameon=False)
    ax.text(
        0.0,
        -0.13,
        (
            "Each head is centered across its 16 rows; the combined matrix is centered "
            "across all 64 rows. No softmax is used."
        ),
        transform=ax.transAxes,
        fontsize=9.5,
        color="#475569",
    )
    fig.savefig(PNG_OUT, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    model = load_model()
    layer = model.layers[0]

    # The observations are the ten digit-token unembedding vectors in 64D.
    digit_unembedding = model.unembed.weight.detach()[:10].double()
    centered_digits = digit_unembedding - digit_unembedding.mean(dim=0, keepdim=True)
    _, digit_singular_values, digit_vh = torch.linalg.svd(
        centered_digits, full_matrices=False
    )
    basis = digit_vh[:3].T  # 64 x 3, with orthonormal columns.

    # O_h is 16 x 64 in row-vector notation: value_h @ O_h -> residual write.
    stored_w_o = layer.W_O.weight.detach().double()
    output_matrices = [
        stored_w_o[:, head_idx * head.d_head : (head_idx + 1) * head.d_head].T
        for head_idx, head in enumerate(layer.heads)
    ]
    output_matrix = torch.cat(output_matrices, dim=0)
    if output_matrix.shape != (64, 64):
        raise AssertionError(f"unexpected output matrix shape: {output_matrix.shape}")
    if not torch.equal(output_matrix, stored_w_o.T):
        raise AssertionError("stacked per-head output matrices do not equal W_O.T")

    digit_variance_fraction = digit_singular_values.square()
    digit_variance_fraction /= digit_variance_fraction.sum()

    matrix_results = {
        f"H{head_idx}": variance_capture(matrix, basis)
        for head_idx, matrix in enumerate(output_matrices)
    }
    matrix_results["All heads"] = variance_capture(output_matrix, basis)

    results = {
        "description": (
            "Centered row variance of each 16x64 per-head output matrix and the combined "
            "64x64 output matrix captured by the top three centered digit-unembedding PCs."
        ),
        "hf_repo": HF_REPO,
        "unembedding_basis": {
            "tokens": list(range(10)),
            "matrix_shape": list(digit_unembedding.shape),
            "centering": "subtract the mean of the ten digit-unembedding rows",
            "basis_shape": list(basis.shape),
            "variance_fraction_by_pc": [
                float(value) for value in digit_variance_fraction[:3]
            ],
            "top3_explained_variance_fraction": float(
                digit_variance_fraction[:3].sum()
            ),
        },
        "output_matrix_orientation": (
            "O_h is 16x64 and value_h @ O_h is a 64D residual write; stacking O_h gives "
            "the 64x64 row-vector map, equal to the stored PyTorch W_O.weight.T."
        ),
        "centering": (
            "Each Hh matrix is centered across its own 16 rows. The All heads matrix is "
            "centered once across all 64 stacked rows."
        ),
        "formula": (
            "fraction_j = q_j^T C_O q_j / trace(C_O), where "
            "C_O = O_centered^T O_centered / (n_rows - 1)."
        ),
        "matrices": matrix_results,
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    plot_results(results)

    print("unembedding_basis,digit-only centered PCA")
    print(f"digit_unembedding_shape,{tuple(digit_unembedding.shape)}")
    print(
        "matrix,shape,total_centered_variance,pc1_fraction,pc2_fraction,"
        "pc3_fraction,top3_fraction,top3_percent"
    )
    for label in ["H0", "H1", "H2", "H3", "All heads"]:
        row = matrix_results[label]
        fractions = row["fraction_by_unembedding_pc"]
        print(
            f"{label},{tuple(row['shape'])},{row['total_centered_variance']:.12f},"
            f"{fractions[0]:.12f},{fractions[1]:.12f},{fractions[2]:.12f},"
            f"{row['top3_captured_fraction']:.12f},{row['top3_captured_percent']:.6f}"
        )
    print(f"wrote,{JSON_OUT}")
    print(f"wrote,{PNG_OUT}")


if __name__ == "__main__":
    main()
