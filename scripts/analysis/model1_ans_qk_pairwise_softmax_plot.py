#!/usr/bin/env python3
"""Plot pairwise digit-key versus ANS-self softmax probabilities by head."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from model1_piecewise_write_animation import load_model


ROOT = Path(__file__).resolve().parents[2]
PNG_OUT = ROOT / "docs" / "assets" / "main_results_ans_qk_pairwise_softmax.png"
PNG_MOBILE_OUT = (
    ROOT / "docs" / "assets" / "main_results_ans_qk_pairwise_softmax_mobile.png"
)
JSON_OUT = ROOT / "docs" / "assets" / "main_results_ans_qk_pairwise_softmax.json"
HEAD_COLORS = ("#2563eb", "#d1495b", "#14866d", "#d97706")
EXPECTED_RECRUITMENT = {
    0: [9],
    1: [],
    2: [7, 8, 9],
    3: [2, 3, 4, 5, 6, 7, 8, 9],
}


def draw_head_panel(ax, row: dict, color: str) -> None:
    x = np.arange(10)
    probabilities = np.asarray(row["pairwise_digit_probabilities"])
    ans_probabilities = 1.0 - probabilities
    ax.set_facecolor("#fafaf8")
    ax.bar(
        x,
        probabilities,
        width=0.72,
        color=color,
        edgecolor="#ffffff",
        linewidth=0.8,
        zorder=3,
    )
    ax.bar(
        x,
        ans_probabilities,
        width=0.72,
        bottom=probabilities,
        color="#e5e7eb",
        edgecolor="#ffffff",
        linewidth=0.8,
        zorder=2,
    )
    ax.axhline(0.5, color="#111827", linewidth=1.1, linestyle="--", zorder=4)
    ax.text(
        9.48,
        0.515,
        "digit beats [ANS]",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#374151",
    )

    recruited = row["digits_above_ans_self"]
    recruited_label = "none" if not recruited else ", ".join(map(str, recruited))
    ax.set_title(f"Head {row['head']}", loc="left", fontsize=14, pad=12)
    ax.text(
        1.0,
        1.025,
        f"number keys above self: {recruited_label}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=color,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in x])
    for tick, digit in zip(ax.get_xticklabels(), x):
        if int(digit) in recruited:
            tick.set_color(color)
            tick.set_fontweight("bold")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 5))
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.65, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, pad=7)


def main() -> None:
    torch.manual_seed(0)
    model, _ = load_model()
    embeddings = model.tok_embed.weight.detach()
    positions = model.pos_embed.weight.detach()
    ans_residual = embeddings[12] + positions[10]
    digit_embeddings = embeddings[:10]

    rows = []
    for head_idx, head in enumerate(model.layers[0].heads):
        query = ans_residual @ head.W_Q.weight.detach().T
        digit_keys = digit_embeddings @ head.W_K.weight.detach().T
        ans_key = ans_residual @ head.W_K.weight.detach().T
        scale = head.d_head**0.5

        digit_scores = query @ digit_keys.T / scale
        ans_self_score = query @ ans_key / scale
        pair_logits = torch.stack(
            [digit_scores, ans_self_score.expand_as(digit_scores)], dim=1
        )
        pair_probabilities = torch.softmax(pair_logits, dim=1)
        digit_probabilities = pair_probabilities[:, 0]
        recruited = [
            digit
            for digit, probability in enumerate(digit_probabilities)
            if float(probability) > 0.5
        ]
        if recruited != EXPECTED_RECRUITMENT[head_idx]:
            raise AssertionError(
                f"unexpected recruitment for H{head_idx}: {recruited}"
            )

        rows.append(
            {
                "head": head_idx,
                "query": "(E[ANS] + P[10]) @ W_Q.T",
                "digit_keys": "E[digit] @ W_K.T",
                "ans_self_key": "(E[ANS] + P[10]) @ W_K.T",
                "score_scale": "1 / sqrt(d_head) = 1 / sqrt(16)",
                "digit_scores": [float(value) for value in digit_scores],
                "ans_self_score": float(ans_self_score),
                "pairwise_digit_probabilities": [
                    float(value) for value in digit_probabilities
                ],
                "pairwise_ans_probabilities": [
                    float(value) for value in pair_probabilities[:, 1]
                ],
                "digits_above_ans_self": recruited,
            }
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "text.color": "#111827",
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.patch.set_facecolor("#ffffff")

    for row, ax, color in zip(rows, axes.flat, HEAD_COLORS):
        draw_head_panel(ax, row, color)

    axes[0, 0].set_ylabel("Pairwise softmax probability")
    axes[1, 0].set_ylabel("Pairwise softmax probability")
    axes[1, 0].set_xlabel("Digit key token")
    axes[1, 1].set_xlabel("Digit key token")

    fig.suptitle(
        "When does a digit key beat the [ANS] self key?",
        x=0.07,
        y=0.975,
        ha="left",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.932,
        (
            "For each digit n: two-way softmax over scaled QK(ANS, n) and "
            "QK(ANS, ANS). The 50% line is the recruitment threshold."
        ),
        ha="left",
        fontsize=10.5,
        color="#4b5563",
    )
    fig.legend(
        handles=[
            Patch(facecolor="#4b83d1", label="digit-key probability"),
            Patch(facecolor="#e5e7eb", label="[ANS]-self probability"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.94, 0.974),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    fig.text(
        0.07,
        0.018,
        (
            "Diagnostic candidate set only: number keys use token embeddings without "
            "source-position embeddings. This is not a complete sequence attention row."
        ),
        ha="left",
        fontsize=9,
        color="#6b7280",
    )
    fig.subplots_adjust(left=0.07, right=0.96, top=0.86, bottom=0.09, hspace=0.30, wspace=0.15)

    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)

    mobile_fig, mobile_axes = plt.subplots(
        4,
        1,
        figsize=(7.2, 13.5),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    mobile_fig.patch.set_facecolor("#ffffff")
    for row, ax, color in zip(rows, mobile_axes, HEAD_COLORS):
        draw_head_panel(ax, row, color)
        ax.set_ylabel("Probability")
    mobile_axes[-1].set_xlabel("Digit key token")
    mobile_fig.suptitle(
        "Digit key vs [ANS] self key",
        x=0.10,
        y=0.987,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    mobile_fig.text(
        0.10,
        0.962,
        "Two-way softmax; crossing 50% means the digit key wins.",
        ha="left",
        fontsize=9.5,
        color="#4b5563",
    )
    mobile_fig.legend(
        handles=[
            Patch(facecolor="#4b83d1", label="digit-key probability"),
            Patch(facecolor="#e5e7eb", label="[ANS]-self probability"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    mobile_fig.subplots_adjust(
        left=0.10,
        right=0.96,
        top=0.92,
        bottom=0.07,
        hspace=0.42,
    )
    mobile_fig.savefig(
        PNG_MOBILE_OUT,
        dpi=190,
        facecolor=mobile_fig.get_facecolor(),
    )
    plt.close(mobile_fig)

    result = {
        "description": (
            "Pairwise softmax diagnostic comparing each token-only digit key with the "
            "ANS@position-10 self key for the ANS query in each head."
        ),
        "softmax_definition": (
            "For each head h and digit n, softmax([score_h(ANS,n), "
            "score_h(ANS,ANS)])."
        ),
        "actual_attention_warning": (
            "This pairwise diagnostic is not an actual attention distribution. Actual "
            "attention normalizes over all sequence positions and number keys include "
            "their source-position embeddings."
        ),
        "rows": rows,
    }
    JSON_OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print("head,digits_above_ans_self")
    for row in rows:
        recruited = row["digits_above_ans_self"]
        print(f"{row['head']},{'+'.join(map(str, recruited)) if recruited else 'none'}")
    print(f"wrote,{PNG_OUT}")
    print(f"wrote,{PNG_MOBILE_OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
