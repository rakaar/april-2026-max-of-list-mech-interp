#!/usr/bin/env python3
"""Plot mean ANS-query attention conditioned on each true maximum."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from model1_piecewise_write_animation import load_model


ROOT = Path(__file__).resolve().parents[2]
PNG_OUT = ROOT / "docs" / "assets" / "main_results_ans_attention_by_max.png"
PNG_MOBILE_OUT = (
    ROOT / "docs" / "assets" / "main_results_ans_attention_by_max_mobile.png"
)
JSON_OUT = ROOT / "docs" / "assets" / "main_results_ans_attention_regimes.json"
BATCH_SIZE = 4096
VOCAB_SIZE = 14
TOKEN_LABELS = [str(value) for value in range(10)] + ["BOS", "SEP", "ANS", "EOS"]
PLOT_TOKEN_LABELS = [str(value) for value in range(10)] + ["B", "S", "A", "E"]
GROUPS = (
    ("max = 0", (0,)),
    ("max = 1", (1,)),
    ("max = 2-6", (2, 3, 4, 5, 6)),
    ("max = 7-8", (7, 8)),
    ("max = 9", (9,)),
)


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    numbers = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)])
    tokens = torch.empty((numbers.shape[0], 11), dtype=torch.long)
    tokens[:, 0] = 10
    tokens[:, 1] = numbers[:, 0]
    tokens[:, 2] = 11
    tokens[:, 3] = numbers[:, 1]
    tokens[:, 4] = 11
    tokens[:, 5] = numbers[:, 2]
    tokens[:, 6] = 11
    tokens[:, 7] = numbers[:, 3]
    tokens[:, 8] = 11
    tokens[:, 9] = numbers[:, 4]
    tokens[:, 10] = 12
    return numbers, tokens


def conditional_token_attention(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Average ANS attention by token identity for each true maximum.

    Attention to repeated occurrences of a token is summed within each input
    before the conditional mean is taken. The resulting rows remain probability
    distributions over the 14 token identities.
    """

    numbers, tokens = make_inputs()
    true_max = numbers.max(dim=1).values
    device = next(model.parameters()).device
    sums = torch.zeros((10, 4, VOCAB_SIZE), dtype=torch.float64)
    counts = torch.zeros(10, dtype=torch.long)

    with torch.no_grad():
        for start in range(0, tokens.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, tokens.shape[0])
            batch_tokens = tokens[start:end].to(device)
            batch_max = true_max[start:end]
            _, attention_patterns = model(batch_tokens)
            ans_attention = attention_patterns[0][:, :, -1, :]

            token_attention = torch.zeros(
                (batch_tokens.shape[0], 4, VOCAB_SIZE),
                dtype=ans_attention.dtype,
                device=device,
            )
            token_indices = batch_tokens[:, None, :].expand(-1, 4, -1)
            token_attention.scatter_add_(2, token_indices, ans_attention)

            sums.index_add_(0, batch_max, token_attention.double().cpu())
            counts += torch.bincount(batch_max, minlength=10)

    expected_counts = torch.tensor(
        [(max_value + 1) ** 5 - max_value**5 for max_value in range(10)]
    )
    if not torch.equal(counts, expected_counts):
        raise AssertionError(f"unexpected conditional counts: {counts.tolist()}")
    if int(counts.sum()) != 100_000:
        raise AssertionError(f"expected 100,000 inputs, found {int(counts.sum())}")

    means = sums / counts[:, None, None]
    if not torch.isfinite(means).all():
        raise AssertionError("conditional attention means contain non-finite values")
    if not torch.allclose(
        means.sum(dim=2), torch.ones((10, 4), dtype=torch.float64), atol=2e-6, rtol=0.0
    ):
        raise AssertionError("conditional token-attention rows do not sum to one")
    if not torch.equal(means[:, :, 13], torch.zeros_like(means[:, :, 13])):
        raise AssertionError("EOS receives attention even though it is not in the input")
    return means, counts


def draw_heatmap(ax, case: dict, compact: bool):
    matrix = np.asarray(case["attention"], dtype=float)
    image = ax.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
        interpolation="nearest",
        cmap=LinearSegmentedColormap.from_list(
            "attention",
            ["#f7f8f6", "#d9eee7", "#5ab69c", "#0b6258"],
        ),
    )

    ax.set_xticks(np.arange(VOCAB_SIZE))
    ax.set_xticklabels(PLOT_TOKEN_LABELS, fontsize=7 if compact else 8)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=False, bottom=False, pad=4)
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(["H0", "H1", "H2", "H3"], fontsize=9 if compact else 10)
    ax.tick_params(axis="y", left=False, pad=7)
    ax.set_title(
        f"max = {case['max_value']}   n = {case['count']:,}",
        loc="left",
        pad=21 if compact else 23,
        fontsize=10 if compact else 12,
        fontweight="bold",
        color="#172126",
    )

    for head_index, top_token_index in enumerate(case["top_token_indices_by_head"]):
        ax.add_patch(
            Rectangle(
                (top_token_index - 0.5, head_index - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#d1495b",
                linewidth=1.8,
            )
        )

    for spine in ax.spines.values():
        spine.set_edgecolor("#cfd8d6")
        spine.set_linewidth(0.8)
    return image


def make_figure(cases: list[dict], compact: bool) -> plt.Figure:
    case_map = {case["max_value"]: case for case in cases}
    max_columns = max(len(max_values) for _, max_values in GROUPS)
    figsize = (8.3, 17.0) if compact else (18.0, 13.5)
    figure, axes = plt.subplots(
        len(GROUPS),
        max_columns,
        figsize=figsize,
        squeeze=False,
        constrained_layout=False,
    )
    figure.patch.set_facecolor("#ffffff")
    image = None
    for row_index, (group_label, max_values) in enumerate(GROUPS):
        for column_index, max_value in enumerate(max_values):
            axis = axes[row_index, column_index]
            image = draw_heatmap(axis, case_map[max_value], compact)
            if column_index == 0:
                axis.set_ylabel(
                    group_label,
                    rotation=0,
                    labelpad=54,
                    ha="right",
                    va="center",
                    fontsize=9.5 if compact else 11,
                    fontweight="bold",
                    color="#263238",
                )

        for column_index in range(len(max_values), max_columns):
            axes[row_index, column_index].axis("off")

    figure.suptitle(
        "Mean [ANS] attention conditioned on the true maximum",
        x=0.075 if compact else 0.08,
        y=0.995 if compact else 0.992,
        ha="left",
        fontsize=16 if compact else 21,
        fontweight="bold",
        color="#111827",
    )
    figure.text(
        0.075 if compact else 0.08,
        0.956,
        "Each matrix is 4 heads x 14 token identities. Repeated occurrences are summed "
        "within each input before averaging over every input with the stated maximum.",
        ha="left",
        fontsize=8.5 if compact else 10,
        color="#58666d",
    )
    figure.text(
        0.075 if compact else 0.08,
        0.014,
        "All 100,000 inputs are included. Coral outlines mark each row's largest mean token mass.",
        ha="left",
        fontsize=8 if compact else 9,
        color="#58666d",
    )
    if image is None:
        raise AssertionError("no attention heatmaps were drawn")

    left = 0.13 if compact else 0.09
    right = 0.965
    colorbar_axis = figure.add_axes([left, 0.06, right - left, 0.012])
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["0%", "50%", "100%"])
    colorbar.set_label("Mean attention probability", color="#58666d")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=8)

    figure.subplots_adjust(
        left=0.13,
        right=0.965,
        top=0.92,
        bottom=0.12,
        hspace=0.65 if compact else 0.7,
        wspace=0.28,
    )
    return figure


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model()
    model = model.to(device)
    means, counts = conditional_token_attention(model)

    individual = []
    for max_value in range(10):
        attention = means[max_value]
        top_token_indices = attention.argmax(dim=1).tolist()
        individual.append(
            {
                "max_value": max_value,
                "count": int(counts[max_value]),
                "token_ids": list(range(VOCAB_SIZE)),
                "token_labels": TOKEN_LABELS,
                "attention": attention.tolist(),
                "top_token_indices_by_head": top_token_indices,
                "top_tokens_by_head": [
                    TOKEN_LABELS[token_index] for token_index in top_token_indices
                ],
            }
        )

    groups = []
    for label, max_values in GROUPS:
        patterns = [individual[max_value]["top_tokens_by_head"] for max_value in max_values]
        groups.append(
            {
                "label": label,
                "max_values": list(max_values),
                "shared_top_tokens_by_head": (
                    patterns[0]
                    if all(pattern == patterns[0] for pattern in patterns[1:])
                    else None
                ),
            }
        )

    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    desktop_figure = make_figure(individual, compact=False)
    desktop_figure.savefig(PNG_OUT, dpi=190, facecolor="#ffffff")
    plt.close(desktop_figure)
    mobile_figure = make_figure(individual, compact=True)
    mobile_figure.savefig(PNG_MOBILE_OUT, dpi=190, facecolor="#ffffff")
    plt.close(mobile_figure)

    result = {
        "description": (
            "Conditional mean final-row softmax attention for the ANS query in each "
            "head, grouped by true maximum and aggregated by token identity."
        ),
        "aggregation": {
            "conditioning": "all five-digit inputs whose true maximum equals d",
            "within_input": (
                "sum attention over all source positions carrying the same token identity"
            ),
            "across_inputs": "arithmetic mean within each true-maximum condition",
        },
        "n_inputs_total": int(counts.sum()),
        "matrix_shape": [4, VOCAB_SIZE],
        "head_axis": ["H0", "H1", "H2", "H3"],
        "token_axis": [
            {"id": token_id, "label": label}
            for token_id, label in enumerate(TOKEN_LABELS)
        ],
        "figure_uses_conditional_means": True,
        "figure_uses_grouped_means": False,
        "individual_cases": individual,
        "groups": groups,
        "validation": {
            "expected_counts_by_true_max": {
                str(max_value): (max_value + 1) ** 5 - max_value**5
                for max_value in range(10)
            },
            "all_100000_inputs_included": True,
            "all_attention_rows_sum_to_one": True,
            "eos_attention_is_zero": True,
        },
    }
    JSON_OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print("true_max,count,H0_top,H1_top,H2_top,H3_top")
    for case in individual:
        print(
            f"{case['max_value']},{case['count']},"
            + ",".join(case["top_tokens_by_head"])
        )
    print(f"wrote,{PNG_OUT}")
    print(f"wrote,{PNG_MOBILE_OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
