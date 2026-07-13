#!/usr/bin/env python3
"""Cosine geometry of source-digit head-sum contributions conditioned on true max."""

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
OUT = ROOT / "docs" / "assets" / "model1_head_sum_source_digit_cosine_by_max.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_head_sum_source_digit_cosine_by_max.json"
BATCH_SIZE = 4096
NUMBER_POSITIONS = [1, 3, 5, 7, 9]
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


def source_digit_contributions(model, tokens: torch.Tensor) -> torch.Tensor:
    """Return summed per-source-number contributions to the ANS head output.

    Output shape is batch x 5 x d_model. Entry [:, k, :] is the sum over heads
    of attention(ANS -> number_position_k) * V(number_position_k) @ W_O_h.T.
    Summing these entries over all source positions plus special-token source
    positions gives the full ANS head-sum output.
    """
    device = tokens.device
    batch = tokens.shape[0]
    seq_len = tokens.shape[1]
    layer = model.layers[0]
    w_o = layer.W_O.weight.detach()
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    resid = model.tok_embed(tokens) + model.pos_embed(positions)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
    source_contrib = torch.zeros(batch, len(NUMBER_POSITIONS), model.d_model, device=device)

    for head_idx, head in enumerate(layer.heads):
        q = head.W_Q(resid)
        k = head.W_K(resid)
        v = head.W_V(resid)
        scores = torch.einsum("bid,bjd->bij", q, k) / (head.d_head**0.5)
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ans_to_numbers = attn[:, 10, NUMBER_POSITIONS]
        values_at_numbers = v[:, NUMBER_POSITIONS, :]
        weighted_values = values_at_numbers * ans_to_numbers.unsqueeze(-1)
        d_head = head.d_head
        w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
        source_contrib += weighted_values @ w_o_head.T

    return source_contrib


def compute_pairwise(
    sum_unit: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    matrices = torch.full((10, 10, 10), float("nan"), dtype=torch.float64)
    for max_value in range(10):
        for row_digit in range(10):
            for col_digit in range(10):
                denom = counts[max_value, row_digit] * counts[max_value, col_digit]
                if int(denom) == 0:
                    continue
                matrices[max_value, row_digit, col_digit] = (
                    sum_unit[max_value, row_digit] @ sum_unit[max_value, col_digit]
                    / denom.to(torch.float64)
                )
    return matrices.clamp(-1.0, 1.0)


def plot_panels(matrices: torch.Tensor, counts: torch.Tensor) -> None:
    plot_matrices = matrices.clone()
    for max_value in range(10):
        for row_digit in range(10):
            for col_digit in range(10):
                if col_digit <= row_digit:
                    plot_matrices[max_value, row_digit, col_digit] = float("nan")

    finite = plot_matrices[~torch.isnan(plot_matrices)]
    max_abs = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#f3f4f6")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5), constrained_layout=True)
    im = None
    for max_value, ax in enumerate(axes.flat):
        matrix = plot_matrices[max_value]
        im = ax.imshow(
            matrix.numpy(),
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
        )
        ax.set_title(f"true max = {max_value}")
        ax.set_xticks(range(10), range(10), fontsize=8)
        ax.set_yticks(range(10), range(10), fontsize=8)
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(9.5, -0.5)
        if max_value == 0:
            ax.text(
                4.5,
                4.5,
                "only digit 0\npresent",
                ha="center",
                va="center",
                fontsize=9,
                color="#374151",
            )
        for row_digit in range(10):
            for col_digit in range(row_digit + 1, 10):
                if int(counts[max_value, row_digit]) == 0 or int(counts[max_value, col_digit]) == 0:
                    continue
                value = float(matrices[max_value, row_digit, col_digit])
                ax.text(
                    col_digit,
                    row_digit,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(value) > 0.55 * max_abs else "black",
                )

    for ax in axes[1, :]:
        ax.set_xlabel("source digit b")
    for ax in axes[:, 0]:
        ax.set_ylabel("source digit a")
    fig.suptitle(
        "Model 1: source-digit contribution cosine by fixed true max",
        fontsize=16,
    )
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012, label="Mean pairwise cosine")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)


def nullable_float(value: torch.Tensor, *, enabled: bool = True) -> float | None:
    if not enabled or torch.isnan(value):
        return None
    return float(value)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model()
    model = model.to(device)
    nums, tokens = make_inputs(device)
    labels = nums.max(dim=1).values

    input_counts = torch.bincount(labels, minlength=10).cpu()
    expected_counts = torch.tensor([EXPECTED_COUNTS_BY_MAX[i] for i in range(10)])
    if not torch.equal(input_counts, expected_counts):
        raise ValueError(f"unexpected true-max counts: {input_counts.tolist()}")

    sum_unit = torch.zeros(10, 10, model.d_model, dtype=torch.float64)
    contribution_counts = torch.zeros(10, 10, dtype=torch.long)
    norm_sum = torch.zeros(10, 10, dtype=torch.float64)

    with torch.no_grad():
        for start in range(0, tokens.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, tokens.shape[0])
            batch_nums = nums[start:end]
            batch_tokens = tokens[start:end]
            batch_labels = labels[start:end]
            contrib = source_digit_contributions(model, batch_tokens)
            contrib_norm = contrib.norm(dim=-1).cpu().to(torch.float64)
            unit = F.normalize(contrib, dim=-1).cpu().to(torch.float64)
            batch_nums_cpu = batch_nums.cpu()
            batch_labels_cpu = batch_labels.cpu()

            for max_value in range(10):
                max_mask = batch_labels_cpu == max_value
                if not bool(max_mask.any()):
                    continue
                for digit in range(10):
                    digit_mask = (batch_nums_cpu == digit) & max_mask[:, None]
                    if not bool(digit_mask.any()):
                        continue
                    selected = unit[digit_mask]
                    selected_norm = contrib_norm[digit_mask]
                    sum_unit[max_value, digit] += selected.sum(dim=0)
                    contribution_counts[max_value, digit] += int(digit_mask.sum())
                    norm_sum[max_value, digit] += selected_norm.sum()

    matrices = compute_pairwise(sum_unit, contribution_counts)
    plot_panels(matrices, contribution_counts)

    data = {
        "description": (
            "All 10^5 Model 1 inputs. For each input and each number position, "
            "source_digit_contribution is the 64d contribution of that source "
            "number to the ANS head-sum output, summed across heads after each "
            "head's W_O slice: attention_h(ANS, source_pos) * V_h(source_pos) "
            "@ W_O_h.T. Contributions are normalized before grouping. For each "
            "fixed true max n, pairwise_avg_cosine_by_true_max[n][a][b] is the "
            "average pairwise cosine between normalized source contributions "
            "from source digit a and source digit b among inputs with true max n. "
            "Duplicate query entries count as repeated source positions."
        ),
        "hf_repo": "andyrdt/04_2026_puzzle_1a",
        "model_config": config,
        "n_inputs_total": int(tokens.shape[0]),
        "sequence_format": "[BOS] n0 [SEP] n1 [SEP] n2 [SEP] n3 [SEP] n4 [ANS]",
        "number_positions": NUMBER_POSITIONS,
        "source_contribution_shape_per_position": [1, model.d_model],
        "input_counts_by_true_max": {str(i): int(input_counts[i]) for i in range(10)},
        "source_position_counts_by_true_max_and_digit": {
            str(max_value): {
                str(digit): int(contribution_counts[max_value, digit])
                for digit in range(10)
            }
            for max_value in range(10)
        },
        "pairwise_avg_cosine_by_true_max": {
            str(max_value): [
                [
                    nullable_float(
                        matrices[max_value, row_digit, col_digit],
                        enabled=bool(
                            contribution_counts[max_value, row_digit] > 0
                            and contribution_counts[max_value, col_digit] > 0
                        ),
                    )
                    for col_digit in range(10)
                ]
                for row_digit in range(10)
            ]
            for max_value in range(10)
        },
        "upper_triangle_off_diagonal_by_true_max": {
            str(max_value): [
                [
                    nullable_float(
                        matrices[max_value, row_digit, col_digit],
                        enabled=bool(
                            col_digit > row_digit
                            and contribution_counts[max_value, row_digit] > 0
                            and contribution_counts[max_value, col_digit] > 0
                        ),
                    )
                    for col_digit in range(10)
                ]
                for row_digit in range(10)
            ]
            for max_value in range(10)
        },
        "source_contribution_norm_mean_by_true_max_and_digit": {
            str(max_value): {
                str(digit): (
                    None
                    if int(contribution_counts[max_value, digit]) == 0
                    else float(norm_sum[max_value, digit] / contribution_counts[max_value, digit])
                )
                for digit in range(10)
            }
            for max_value in range(10)
        },
    }
    JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")

    print("true_max,source_digit_a,source_digit_b,avg_cosine")
    for max_value in range(10):
        for row_digit in range(10):
            for col_digit in range(row_digit + 1, 10):
                if int(contribution_counts[max_value, row_digit]) == 0:
                    continue
                if int(contribution_counts[max_value, col_digit]) == 0:
                    continue
                print(
                    f"{max_value},{row_digit},{col_digit},"
                    f"{float(matrices[max_value, row_digit, col_digit]):.6f}"
                )
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
