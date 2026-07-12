#!/usr/bin/env python3
"""Plot the exact ANS-query attention rows for every possible maximum."""

from __future__ import annotations

import json
from pathlib import Path

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
GROUPS = (
    ("max = 0", (0,)),
    ("max = 1", (1,)),
    ("max = 2-6", (2, 3, 4, 5, 6)),
    ("max = 7-8", (7, 8)),
    ("max = 9", (9,)),
)
EXPECTED_TOP_POSITIONS = {
    0: [10, 10, 10, 10],
    1: [10, 10, 10, 10],
    2: [10, 10, 10, 5],
    3: [10, 10, 10, 5],
    4: [10, 10, 10, 5],
    5: [10, 10, 10, 5],
    6: [10, 10, 10, 5],
    7: [10, 10, 5, 5],
    8: [10, 10, 5, 5],
    9: [5, 10, 5, 5],
}


def tokenize(max_value: int) -> torch.Tensor:
    return torch.tensor(
        [[10, 0, 11, 0, 11, max_value, 11, 0, 11, 0, 12]],
        dtype=torch.long,
    )


def token_labels(max_label: str) -> list[str]:
    return [
        "BOS\n0",
        "0\n1",
        "SEP\n2",
        "0\n3",
        "SEP\n4",
        f"{max_label}\n5",
        "SEP\n6",
        "0\n7",
        "SEP\n8",
        "0\n9",
        "ANS\n10",
    ]


def ans_attention_rows(model, max_value: int) -> torch.Tensor:
    tokens = tokenize(max_value)
    positions = torch.arange(tokens.shape[1]).unsqueeze(0)
    residual = model.tok_embed(tokens) + model.pos_embed(positions)
    mask = torch.tril(torch.ones(tokens.shape[1], tokens.shape[1])).unsqueeze(0)
    rows = []
    with torch.no_grad():
        for head in model.layers[0].heads:
            _, attention = head(residual, mask)
            rows.append(attention[0, -1].detach().cpu())
    return torch.stack(rows)


def draw_heatmap(ax, case: dict, compact: bool) -> None:
    matrix = np.asarray(case["attention"], dtype=float)
    image = ax.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
        cmap=LinearSegmentedColormap.from_list(
            "attention",
            ["#f7f8f6", "#d9eee7", "#5ab69c", "#0b6258"],
        ),
    )

    labels = token_labels(str(case["max_value"]))
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels(labels, fontsize=7.5 if compact else 8.5, linespacing=1.25)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=False, bottom=False, pad=4)
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(["H0", "H1", "H2", "H3"], fontsize=9 if compact else 10)
    ax.tick_params(axis="y", left=False, pad=7)
    ax.set_title(
        f"max = {case['max_value']}",
        loc="left",
        pad=22 if compact else 24,
        fontsize=12 if compact else 14,
        fontweight="bold",
        color="#172126",
    )

    for head_idx in range(4):
        top_position = int(np.argmax(matrix[head_idx]))
        ax.add_patch(
            Rectangle(
                (top_position - 0.5, head_idx - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#d1495b",
                linewidth=2.0,
            )
        )

    for spine in ax.spines.values():
        spine.set_edgecolor("#cfd8d6")
        spine.set_linewidth(0.8)
    return image


def make_figure(cases: list[dict], compact: bool) -> plt.Figure:
    case_map = {case["max_value"]: case for case in cases}
    max_cols = max(len(max_values) for _, max_values in GROUPS)
    figsize = (7.4, 17.0) if compact else (17.6, 16.5)
    fig, axes = plt.subplots(
        len(GROUPS),
        max_cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=False,
    )
    fig.patch.set_facecolor("#ffffff")
    image = None
    for row_idx, (_, max_values) in enumerate(GROUPS):
        for col_idx, max_value in enumerate(max_values):
            ax = axes[row_idx, col_idx]
            image = draw_heatmap(ax, case_map[max_value], compact)
            if col_idx == 0:
                label = list(GROUPS[row_idx])[0]
                ax.set_ylabel(
                    label,
                    rotation=0,
                    labelpad=54,
                    ha="right",
                    va="center",
                    fontsize=11 if not compact else 9.5,
                    fontweight="bold",
                    color="#263238",
                )

        for col_idx in range(len(max_values), max_cols):
            axes[row_idx, col_idx].axis("off")

    title = (
        "Actual [ANS] attention\nfor every maximum"
        if compact
        else "Actual [ANS] attention for every maximum"
    )
    fig.suptitle(
        title,
        x=0.075 if compact else 0.08,
        y=0.995 if compact else 0.992,
        ha="left",
        fontsize=17 if compact else 22,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.075 if compact else 0.08,
        0.954,
        "Ten exact matrices; each is the final softmax row: 4 heads x 11 source tokens. "
        "Rows are grouped as max=0, max=1, max=2-6, max=7-8, max=9.",
        ha="left",
        fontsize=9 if compact else 10.5,
        color="#58666d",
    )
    fig.text(
        0.075 if compact else 0.08,
        0.014,
        (
            "Inputs are [0, 0, m, 0, 0]. No cases are averaged.\n"
            "Coral outlines mark each head's highest-attended source."
            if compact
            else "Inputs are [0, 0, m, 0, 0]. Every matrix is an exact model attention "
            "distribution; no cases are averaged."
        ),
        ha="left",
        fontsize=8 if compact else 9,
        color="#58666d",
    )
    if not compact:
        fig.text(
            0.94,
            0.978,
            "coral outline = row maximum",
            ha="right",
            fontsize=9,
            color="#d1495b",
            fontweight="bold",
        )
    if image is None:
        raise AssertionError("no attention heatmaps were drawn")
    left = 0.11 if compact else 0.09
    right = 0.965
    colorbar_axis = fig.add_axes([left, 0.062 if compact else 0.058, right - left, 0.011])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["0%", "50%", "100%"])
    colorbar.set_label("Attention probability", color="#58666d")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=8)

    fig.subplots_adjust(
        left=0.13,
        right=0.965,
        top=0.922 if compact else 0.928,
        bottom=0.115,
        hspace=0.52 if compact else 0.58,
        wspace=0.22,
    )
    return fig


def main() -> None:
    torch.manual_seed(0)
    model, _ = load_model()

    individual = {}
    for max_value in range(10):
        attention = ans_attention_rows(model, max_value)
        if not torch.allclose(
            attention.sum(dim=1), torch.ones(4), rtol=0.0, atol=1e-6
        ):
            raise AssertionError(f"attention rows do not sum to one for max {max_value}")
        top_positions = attention.argmax(dim=1).tolist()
        if top_positions != EXPECTED_TOP_POSITIONS[max_value]:
            raise AssertionError(
                f"unexpected top sources for max {max_value}: {top_positions}"
            )
        individual[max_value] = {
            "max_value": max_value,
            "numbers": [0, 0, max_value, 0, 0],
            "tokens": tokenize(max_value)[0].tolist(),
            "attention": [
                [float(value) for value in head_row] for head_row in attention
            ],
            "top_positions_by_head": top_positions,
            "top_tokens_by_head": [
                "ANS" if position == 10 else str(max_value)
                for position in top_positions
            ],
        }

    groups = []
    for label, max_values in GROUPS:
        stack = torch.stack(
            [
                torch.tensor(individual[max_value]["attention"])
                for max_value in max_values
            ]
        )
        mean_attention = stack.mean(dim=0)
        member_patterns = [
            individual[max_value]["top_positions_by_head"]
            for max_value in max_values
        ]
        if any(pattern != member_patterns[0] for pattern in member_patterns[1:]):
            raise AssertionError(f"group {label} does not share one top-source pattern")
        groups.append(
            {
                "label": label,
                "max_values": list(max_values),
                "max_token_label": (
                    str(max_values[0])
                    if len(max_values) == 1
                    else f"{max_values[0]}-{max_values[-1]}"
                ),
                "mean_attention": [
                    [float(value) for value in head_row]
                    for head_row in mean_attention
                ],
                "shared_top_positions_by_head": member_patterns[0],
                "shared_top_sources_by_head": [
                    "ANS" if position == 10 else "max"
                    for position in member_patterns[0]
                ],
            }
        )

    cases = [
        {
            "max_value": max_value,
            "attention": individual[max_value]["attention"],
        }
        for max_value in range(10)
    ]

    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    desktop_figure = make_figure(cases, compact=False)
    desktop_figure.savefig(PNG_OUT, dpi=190, facecolor="#ffffff")
    plt.close(desktop_figure)
    mobile_figure = make_figure(cases, compact=True)
    mobile_figure.savefig(PNG_MOBILE_OUT, dpi=190, facecolor="#ffffff")
    plt.close(mobile_figure)

    result = {
        "description": (
            "Actual final-row softmax attention for the ANS query in each head, "
            "using matched inputs [0, 0, m, 0, 0]."
        ),
        "matrix_shape": [4, 11],
        "head_axis": ["H0", "H1", "H2", "H3"],
        "source_positions": list(range(11)),
        "figure_uses_grouped_means": False,
        "individual_cases": [individual[max_value] for max_value in range(10)],
        "groups": groups,
        "validation": {
            "all_attention_rows_sum_to_one": True,
            "all_members_of_each_group_share_top_source_pattern": True,
        },
    }
    JSON_OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print("regime,H0,H1,H2,H3")
    for group in groups:
        print(f"{group['label']}," + ",".join(group["shared_top_sources_by_head"]))
    print(f"wrote,{PNG_OUT}")
    print(f"wrote,{PNG_MOBILE_OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
