#!/usr/bin/env python3
"""Decompose the max-8/max-9 boundary on selected examples."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
MARGIN_OUT = ROOT / "docs" / "assets" / "model1_margin_9v8_examples.png"
MARGIN_JSON_OUT = ROOT / "docs" / "assets" / "model1_margin_9v8_examples.json"
TARGET_OUT = ROOT / "docs" / "assets" / "model1_target_logit_contribution_8v9_examples.png"
TARGET_JSON_OUT = ROOT / "docs" / "assets" / "model1_target_logit_contribution_8v9_examples.json"

EXAMPLES = [
    [8, 2, 3, 4, 5],
    [2, 8, 4, 7, 3],
    [2, 3, 8, 4, 7],
    [7, 6, 5, 4, 8],
    [8, 8, 7, 6, 5],
    [9, 2, 3, 4, 5],
    [2, 9, 4, 8, 3],
    [2, 3, 9, 4, 8],
    [8, 7, 6, 5, 9],
    [9, 9, 8, 7, 6],
]

NUMBER_POSITIONS = [1, 3, 5, 7, 9]
COMPONENTS = ["residual", "H0", "H1", "H2", "H3"]
PLOT_COMPONENTS = COMPONENTS + ["final"]


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


def tokenize(nums: list[int]) -> list[int]:
    return [10, nums[0], 11, nums[1], 11, nums[2], 11, nums[3], 11, nums[4], 12]


def number_logits(vec: torch.Tensor, model) -> torch.Tensor:
    return vec @ model.unembed.weight.detach()[:10].T


def decompose(model):
    tokens = torch.tensor([tokenize(nums) for nums in EXAMPLES], dtype=torch.long)
    nums_t = torch.tensor(EXAMPLES, dtype=torch.long)
    labels = nums_t.max(dim=1).values

    with torch.no_grad():
        model_logits, _ = model(tokens)
        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len).unsqueeze(0)
        resid = model.tok_embed(tokens) + model.pos_embed(positions)
        ans_resid = resid[:, 10, :]
        layer = model.layers[0]
        w_o = layer.W_O.weight.detach()
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

        head_vectors = []
        head_attn_rows = []
        for head_idx, head in enumerate(layer.heads):
            head_values, attn = head(resid, mask)
            d_head = head.d_head
            w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
            head_vectors.append(head_values[:, 10, :] @ w_o_head.T)
            head_attn_rows.append(attn[:, 10, :])

        component_vectors = {
            "residual": ans_resid,
            "H0": head_vectors[0],
            "H1": head_vectors[1],
            "H2": head_vectors[2],
            "H3": head_vectors[3],
        }
        component_vectors["final"] = ans_resid + sum(head_vectors)
        component_logits = {
            name: number_logits(vec, model) for name, vec in component_vectors.items()
        }

    return tokens, nums_t, labels, model_logits, component_logits, head_attn_rows


def write_margin_result(tokens, nums_t, labels, model_logits, component_logits, head_attn_rows):
    margins_9v8 = {
        name: logits[:, 9] - logits[:, 8] for name, logits in component_logits.items()
    }

    rows = []
    for idx, nums in enumerate(EXAMPLES):
        max_value = int(labels[idx])
        row = {
            "example_index": idx,
            "nums": nums,
            "true_max": max_value,
            "model_pred": int(model_logits[idx, 10, :10].argmax()),
            "component_logits_8_9": {
                name: {
                    "logit8": float(component_logits[name][idx, 8]),
                    "logit9": float(component_logits[name][idx, 9]),
                    "margin_9_minus_8": float(margins_9v8[name][idx]),
                }
                for name in PLOT_COMPONENTS
            },
            "attention": {},
        }
        is_max = nums_t[idx] == max_value
        for head_idx, attn_row in enumerate(head_attn_rows):
            number_attn = attn_row[idx, NUMBER_POSITIONS]
            max_mass = float((number_attn * is_max.float()).sum())
            top_pos = int(attn_row[idx].argmax())
            top_token = int(tokens[idx, top_pos])
            row["attention"][f"H{head_idx}"] = {
                "top_position": top_pos,
                "top_token": top_token,
                "top_is_ans": top_pos == 10,
                "max_token_attention_mass": max_mass,
                "ans_self_attention_mass": float(attn_row[idx, 10]),
            }
        rows.append(row)

    summary_by_true_max = {}
    for true_max in [8, 9]:
        indices = [i for i, row in enumerate(rows) if row["true_max"] == true_max]
        summary_by_true_max[str(true_max)] = {
            name: float(torch.stack([margins_9v8[name][i] for i in indices]).mean())
            for name in PLOT_COMPONENTS
        }

    result = {
        "description": "Component decomposition of logit[9] - logit[8] on selected examples.",
        "components": PLOT_COMPONENTS,
        "examples": rows,
        "avg_margin_9_minus_8_by_true_max": summary_by_true_max,
    }
    MARGIN_JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    MARGIN_JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    margin_matrix = torch.stack([margins_9v8[name] for name in PLOT_COMPONENTS])
    fig, ax = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
    vmax = max(abs(float(margin_matrix.min())), abs(float(margin_matrix.max())))
    im = ax.imshow(margin_matrix.detach().numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    labels_x = [f"{i}: {nums}\nmax={int(labels[i])}" for i, nums in enumerate(EXAMPLES)]
    ax.set_xticks(range(len(EXAMPLES)))
    ax.set_xticklabels(labels_x, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(PLOT_COMPONENTS)))
    ax.set_yticklabels(PLOT_COMPONENTS)
    ax.set_title("Component contributions to logit[9] - logit[8]")
    for y in range(len(PLOT_COMPONENTS)):
        for x in range(len(EXAMPLES)):
            value = float(margin_matrix[y, x])
            ax.text(
                x,
                y,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 * vmax else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="logit[9] - logit[8]")
    fig.savefig(MARGIN_OUT, dpi=180)
    plt.close(fig)

    print("avg_margin_9_minus_8_by_true_max")
    for true_max, values in summary_by_true_max.items():
        print(
            true_max + ","
            + ",".join(f"{name}:{values[name]:+.6f}" for name in PLOT_COMPONENTS)
        )
    print("attention_summary")
    for row in rows:
        print(
            f"{row['example_index']},max={row['true_max']},"
            + ",".join(
                f"{head}:top={item['top_position']}:{item['top_token']},"
                f"maxmass={item['max_token_attention_mass']:.3f},"
                f"self={item['ans_self_attention_mass']:.3f}"
                for head, item in row["attention"].items()
            )
        )
    print(f"wrote,{MARGIN_OUT}")
    print(f"wrote,{MARGIN_JSON_OUT}")
    return summary_by_true_max


def write_target_result(labels, model_logits, component_logits):
    batch = len(EXAMPLES)
    raw_target = torch.empty((len(PLOT_COMPONENTS), batch))
    signed_share = torch.empty((len(COMPONENTS), batch))
    positive_share = torch.empty((len(COMPONENTS), batch))

    examples = []
    for idx, nums in enumerate(EXAMPLES):
        target = int(labels[idx])
        final_target = float(component_logits["final"][idx, target])
        component_values = torch.tensor(
            [float(component_logits[name][idx, target]) for name in COMPONENTS]
        )
        positive = component_values.clamp_min(0)
        positive_total = float(positive.sum())

        for row_idx, name in enumerate(PLOT_COMPONENTS):
            raw_target[row_idx, idx] = float(component_logits[name][idx, target])
        for row_idx, name in enumerate(COMPONENTS):
            value = float(component_logits[name][idx, target])
            signed_share[row_idx, idx] = value / final_target if abs(final_target) > 1e-12 else float("nan")
            positive_share[row_idx, idx] = (
                max(value, 0.0) / positive_total if positive_total > 1e-12 else float("nan")
            )

        examples.append(
            {
                "example_index": idx,
                "nums": nums,
                "target_logit": target,
                "model_pred": int(model_logits[idx, 10, :10].argmax()),
                "raw_target_logit_contribution": {
                    name: float(component_logits[name][idx, target]) for name in PLOT_COMPONENTS
                },
                "signed_share_of_final_target_logit": {
                    name: float(signed_share[row_idx, idx])
                    for row_idx, name in enumerate(COMPONENTS)
                },
                "positive_share_of_positive_target_logit_contributions": {
                    name: float(positive_share[row_idx, idx])
                    for row_idx, name in enumerate(COMPONENTS)
                },
            }
        )

    averages = {}
    for target in [8, 9]:
        indices = [idx for idx, label in enumerate(labels.tolist()) if label == target]
        averages[str(target)] = {
            "raw_target_logit_contribution": {
                name: float(torch.tensor([component_logits[name][idx, target] for idx in indices]).mean())
                for name in PLOT_COMPONENTS
            },
            "signed_share_of_final_target_logit": {
                name: float(torch.stack([signed_share[row_idx, idx] for idx in indices]).mean())
                for row_idx, name in enumerate(COMPONENTS)
            },
            "positive_share_of_positive_target_logit_contributions": {
                name: float(torch.stack([positive_share[row_idx, idx] for idx in indices]).mean())
                for row_idx, name in enumerate(COMPONENTS)
            },
        }

    result = {
        "description": (
            "For max-8 examples, target logit is 8. For max-9 examples, target logit is 9. "
            "Raw contributions are additive. Signed shares can be negative or exceed 1 due to "
            "cancellation. Positive shares sum to 1 across positive contributors only."
        ),
        "components": COMPONENTS,
        "examples": examples,
        "averages_by_target_logit": averages,
    }
    TARGET_JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    TARGET_JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    labels_x = [f"{i}: {nums}\ntarget={int(labels[i])}" for i, nums in enumerate(EXAMPLES)]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

    vmax = max(abs(float(raw_target.min())), abs(float(raw_target.max())))
    im = axes[0].imshow(raw_target.numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_title("Raw contribution to target logit")
    axes[0].set_yticks(range(len(PLOT_COMPONENTS)))
    axes[0].set_yticklabels(PLOT_COMPONENTS)
    axes[0].set_xticks(range(batch))
    axes[0].set_xticklabels(labels_x, rotation=35, ha="right", fontsize=8)
    for y in range(len(PLOT_COMPONENTS)):
        for x in range(batch):
            value = float(raw_target[y, x])
            axes[0].text(
                x,
                y,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 * vmax else "black",
            )
    fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02, label="target logit contribution")

    im = axes[1].imshow(100 * signed_share.numpy(), cmap="coolwarm", vmin=-150, vmax=150, aspect="auto")
    axes[1].set_title("Signed share of final target logit (%)")
    axes[1].set_yticks(range(len(COMPONENTS)))
    axes[1].set_yticklabels(COMPONENTS)
    axes[1].set_xticks(range(batch))
    axes[1].set_xticklabels(labels_x, rotation=35, ha="right", fontsize=8)
    for y in range(len(COMPONENTS)):
        for x in range(batch):
            value = float(100 * signed_share[y, x])
            axes[1].text(x, y, f"{value:+.0f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.025, pad=0.02, label="% of final target logit")

    im = axes[2].imshow(100 * positive_share.numpy(), cmap="viridis", vmin=0, vmax=100, aspect="auto")
    axes[2].set_title("Positive share among positive target-logit contributors (%)")
    axes[2].set_yticks(range(len(COMPONENTS)))
    axes[2].set_yticklabels(COMPONENTS)
    axes[2].set_xticks(range(batch))
    axes[2].set_xticklabels(labels_x, rotation=35, ha="right", fontsize=8)
    for y in range(len(COMPONENTS)):
        for x in range(batch):
            value = float(100 * positive_share[y, x])
            axes[2].text(
                x,
                y,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 55 else "black",
            )
    fig.colorbar(im, ax=axes[2], fraction=0.025, pad=0.02, label="% of positive contributions")

    fig.suptitle("Model 1: target-logit contribution decomposition for max 8 vs max 9 examples")
    fig.savefig(TARGET_OUT, dpi=180)
    plt.close(fig)

    print("averages_by_target")
    for target, values in averages.items():
        raw = values["raw_target_logit_contribution"]
        pos = values["positive_share_of_positive_target_logit_contributions"]
        print(
            f"target={target},raw="
            + ",".join(f"{name}:{raw[name]:+.6f}" for name in PLOT_COMPONENTS)
        )
        print(
            f"target={target},positive_share="
            + ",".join(f"{name}:{pos[name]:+.6f}" for name in COMPONENTS)
        )
    print(f"wrote,{TARGET_OUT}")
    print(f"wrote,{TARGET_JSON_OUT}")
    return averages


def main() -> None:
    model = load_model()
    tokens, nums_t, labels, model_logits, component_logits, head_attn_rows = decompose(model)
    write_margin_result(tokens, nums_t, labels, model_logits, component_logits, head_attn_rows)
    write_target_result(labels, model_logits, component_logits)


if __name__ == "__main__":
    main()
