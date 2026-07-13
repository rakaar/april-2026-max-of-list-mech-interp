#!/usr/bin/env python3
"""Pairwise cosine between Model 1 head-sum vectors grouped by true max."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_head_sum_pairwise_max_cosine.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_head_sum_pairwise_max_cosine.json"
BATCH_SIZE = 4096
EXPECTED_COUNTS_BY_MAX = {
    0: 1,
    1: 31,
    2: 211,
    3: 781,
    4: 2101,
    5: 4651,
    6: 9031,
    7: 15961,
    8: 26281,
    9: 40951,
}


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


def make_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nums = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)]).to(device)
    tokens = torch.empty((nums.shape[0], 11), dtype=torch.long, device=device)
    tokens[:, 0] = 10
    tokens[:, 1] = nums[:, 0]
    tokens[:, 2] = 11
    tokens[:, 3] = nums[:, 1]
    tokens[:, 4] = 11
    tokens[:, 5] = nums[:, 2]
    tokens[:, 6] = 11
    tokens[:, 7] = nums[:, 3]
    tokens[:, 8] = 11
    tokens[:, 9] = nums[:, 4]
    tokens[:, 10] = 12
    return nums, tokens


def compute_head_sum(model, tokens: torch.Tensor) -> torch.Tensor:
    device = tokens.device
    seq_len = tokens.shape[1]
    layer = model.layers[0]
    w_o = layer.W_O.weight.detach()
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    resid = model.tok_embed(tokens) + model.pos_embed(positions)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)

    head_vectors = []
    for head_idx, head in enumerate(layer.heads):
        head_values, _ = head(resid, causal_mask)
        d_head = head.d_head
        w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
        head_vectors.append(head_values[:, 10, :] @ w_o_head.T)

    return sum(head_vectors)


def plot_upper_triangle(avg_cosine: torch.Tensor) -> None:
    max_values = list(range(10))
    plot_matrix = avg_cosine.clone()
    for row in range(10):
        for col in range(10):
            if col <= row:
                plot_matrix[row, col] = float("nan")

    finite = plot_matrix[~torch.isnan(plot_matrix)]
    max_abs = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#f3f4f6")

    fig, ax = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    im = ax.imshow(
        plot_matrix.numpy(),
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
    )
    ax.set_xticks(max_values, max_values)
    ax.set_yticks(max_values, max_values)
    ax.set_xlabel("True max group m")
    ax.set_ylabel("True max group n")
    ax.set_title("Model 1: pairwise cosine between head-sum vectors by true max")

    for row in range(10):
        for col in range(row + 1, 10):
            value = float(avg_cosine[row, col])
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 * max_abs else "black",
            )

    ax.text(
        0.01,
        -0.08,
        "Cells show mean pairwise cosine between normalized 64d head-sum vectors.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean pairwise cosine")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)


def nullable_float(value: torch.Tensor | float, *, enabled: bool = True) -> float | None:
    if not enabled:
        return None
    return float(value)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model()
    model = model.to(device)
    nums, tokens = make_inputs(device)
    labels = nums.max(dim=1).values

    counts = torch.bincount(labels, minlength=10).cpu()
    expected_counts = torch.tensor([EXPECTED_COUNTS_BY_MAX[i] for i in range(10)])
    if not torch.equal(counts, expected_counts):
        raise ValueError(f"unexpected true-max counts: {counts.tolist()}")

    sum_unit = torch.zeros(10, model.d_model, dtype=torch.float64)
    norm_min = torch.full((10,), float("inf"), dtype=torch.float64)
    norm_max = torch.full((10,), float("-inf"), dtype=torch.float64)
    norm_sum = torch.zeros(10, dtype=torch.float64)

    with torch.no_grad():
        for start in range(0, tokens.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, tokens.shape[0])
            batch_tokens = tokens[start:end]
            batch_labels = labels[start:end]
            batch_labels_cpu = batch_labels.cpu()
            head_sum = compute_head_sum(model, batch_tokens)
            head_norm = head_sum.norm(dim=1).cpu().to(torch.float64)
            unit = F.normalize(head_sum, dim=1).cpu().to(torch.float64)

            for max_value in range(10):
                mask = batch_labels_cpu == max_value
                if not bool(mask.any()):
                    continue
                selected_unit = unit[mask]
                selected_norm = head_norm[mask]
                sum_unit[max_value] += selected_unit.sum(dim=0)
                norm_sum[max_value] += selected_norm.sum()
                norm_min[max_value] = torch.minimum(norm_min[max_value], selected_norm.min())
                norm_max[max_value] = torch.maximum(norm_max[max_value], selected_norm.max())

    counts_f = counts.to(torch.float64)
    avg_cosine = torch.empty(10, 10, dtype=torch.float64)
    for row in range(10):
        for col in range(10):
            avg_cosine[row, col] = (
                sum_unit[row] @ sum_unit[col] / (counts_f[row] * counts_f[col])
            )
    avg_cosine = avg_cosine.clamp(-1.0, 1.0)

    within_group = torch.full((10,), float("nan"), dtype=torch.float64)
    for max_value in range(10):
        count = int(counts[max_value])
        if count > 1:
            within_group[max_value] = torch.clamp(
                ((sum_unit[max_value] @ sum_unit[max_value]) - count)
                / (count * (count - 1)),
                -1.0,
                1.0,
            )

    if not torch.allclose(avg_cosine, avg_cosine.T, atol=1e-10):
        raise ValueError("pairwise cosine matrix is unexpectedly asymmetric")
    if bool((avg_cosine.abs() > 1.000001).any()):
        raise ValueError("pairwise cosine value outside [-1, 1]")

    data = {
        "description": (
            "All 10^5 Model 1 inputs. For each input, head_sum is the actual "
            "ANS-position 64d output H0_vec + H1_vec + H2_vec + H3_vec after each "
            "head's W_O slice. Rows are normalized before grouping. For n != m, "
            "pairwise_avg_cosine[n][m] is the average cosine over every ordered "
            "pair of normalized head-sum vectors from true-max groups n and m. "
            "within_group_excluding_self reports the average within a true-max "
            "group excluding self-pairs."
        ),
        "hf_repo": "andyrdt/04_2026_puzzle_1a",
        "model_config": config,
        "n_inputs_total": int(tokens.shape[0]),
        "sequence_format": "[BOS] n0 [SEP] n1 [SEP] n2 [SEP] n3 [SEP] n4 [ANS]",
        "head_sum_shape_per_input": [1, model.d_model],
        "counts_by_true_max": {str(i): int(counts[i]) for i in range(10)},
        "pairwise_avg_cosine": [
            [float(avg_cosine[row, col]) for col in range(10)] for row in range(10)
        ],
        "upper_triangle_off_diagonal": [
            [
                nullable_float(avg_cosine[row, col], enabled=col > row)
                for col in range(10)
            ]
            for row in range(10)
        ],
        "within_group_excluding_self": {
            str(i): nullable_float(within_group[i], enabled=not torch.isnan(within_group[i]))
            for i in range(10)
        },
        "head_sum_norm_by_true_max": {
            str(i): {
                "mean": float(norm_sum[i] / counts_f[i]),
                "min": float(norm_min[i]),
                "max": float(norm_max[i]),
            }
            for i in range(10)
        },
    }

    JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")
    plot_upper_triangle(avg_cosine)

    print("pairwise_avg_cosine_upper_triangle")
    print("row_true_max,col_true_max,avg_cosine")
    for row in range(10):
        for col in range(row + 1, 10):
            print(f"{row},{col},{float(avg_cosine[row, col]):.6f}")
    print("within_group_excluding_self")
    for max_value in range(10):
        value = "null" if torch.isnan(within_group[max_value]) else f"{float(within_group[max_value]):.6f}"
        print(f"{max_value},{value}")
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
