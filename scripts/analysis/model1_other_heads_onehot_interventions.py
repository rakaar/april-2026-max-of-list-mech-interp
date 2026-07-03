#!/usr/bin/env python3
"""Test one-hot ANS-row replacements for heads 0/1/2/3 on all >=2 inputs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_other_heads_onehot_interventions.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_other_heads_onehot_interventions.json"
BATCH_SIZE = 4096
NUMBER_POSITIONS = torch.tensor([1, 3, 5, 7, 9])
BOS_SEP_POSITIONS = torch.tensor([0, 2, 4, 6, 8])


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


def make_inputs(device: str):
    nums = torch.cartesian_prod(*[torch.arange(2, 10) for _ in range(5)]).to(device)
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


def digit_logits(vec: torch.Tensor, model) -> torch.Tensor:
    return vec @ model.unembed.weight.detach()[:10].T


def add_accuracy(
    counts: dict[str, dict[str, int]],
    per_max_correct: dict[str, torch.Tensor],
    per_max_total: torch.Tensor,
    name: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    pred = logits.argmax(dim=1)
    counts[name]["correct"] += int((pred == labels).sum())
    counts[name]["total"] += int(labels.numel())
    for true_max in range(2, 10):
        mask = labels == true_max
        per_max_correct[name][true_max] += int((pred[mask] == labels[mask]).sum())
        if name == "actual":
            per_max_total[true_max] += int(mask.sum())


def summarize_top_categories(
    top_key_category_counts: dict[str, torch.Tensor],
    counts_by_max: torch.Tensor,
) -> dict[str, dict[str, dict[str, float]]]:
    categories = ["max number", "nonmax number", "ANS self", "BOS/SEP"]
    result = {}
    for name, tensor in top_key_category_counts.items():
        result[name] = {}
        for true_max in range(2, 10):
            denom = max(int(counts_by_max[true_max]), 1)
            result[name][str(true_max)] = {
                category: float(tensor[true_max, idx] / denom)
                for idx, category in enumerate(categories)
            }
    return result


@torch.no_grad()
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device)
    nums, tokens = make_inputs(device)
    labels_all = nums.max(dim=1).values
    total = tokens.shape[0]
    number_positions = NUMBER_POSITIONS.to(device)
    bos_sep_positions = BOS_SEP_POSITIONS.to(device)

    condition_names = [
        "actual",
        "H3 max one-hot",
        "H0/H2 top + H3 max one-hot",
        "all heads top one-hot",
        "H0/H2/H3 max one-hot",
        "all heads max one-hot",
        "only H3 max one-hot",
    ]
    counts = {name: {"correct": 0, "total": 0} for name in condition_names}
    per_max_correct = {name: torch.zeros(10, dtype=torch.long) for name in condition_names}
    per_max_total = torch.zeros(10, dtype=torch.long)

    counts_by_max = torch.zeros(10, dtype=torch.long)
    mass_sums = {
        "max number": torch.zeros(4, 10),
        "nonmax number": torch.zeros(4, 10),
        "ANS self": torch.zeros(4, 10),
        "BOS/SEP": torch.zeros(4, 10),
    }
    top_category_counts = {
        f"H{head}": torch.zeros(10, 4, dtype=torch.long) for head in range(4)
    }

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_nums = nums[start:end]
        labels = labels_all[start:end]
        batch = end - start
        is_max_slot = batch_nums == labels[:, None]
        batch_idx = torch.arange(batch, device=device)

        seq_len = tokens.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        resid = model.tok_embed(tokens[start:end]) + model.pos_embed(positions)
        ans_resid = resid[:, 10, :]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)

        layer = model.layers[0]
        actual_values = []
        top_values = []
        max_values = []
        self_values = []
        zero_values = []
        attn_rows = []

        for head_idx, head in enumerate(layer.heads):
            out, attn = head(resid, mask)
            attn_row = attn[:, 10, :]
            source_values = resid @ head.W_V.weight.detach().T
            number_attn = attn_row[:, number_positions]

            top_pos = attn_row.argmax(dim=1)
            max_attn = number_attn.masked_fill(~is_max_slot, -1.0)
            max_slot = max_attn.argmax(dim=1)
            max_pos = number_positions[max_slot]

            actual_values.append(out[:, 10, :])
            top_values.append(source_values[batch_idx, top_pos])
            max_values.append(source_values[batch_idx, max_pos])
            self_values.append(source_values[:, 10, :])
            zero_values.append(torch.zeros_like(out[:, 10, :]))
            attn_rows.append(attn_row)

            max_mass = (number_attn * is_max_slot.float()).sum(dim=1)
            nonmax_number_mass = (number_attn * (~is_max_slot).float()).sum(dim=1)
            self_mass = attn_row[:, 10]
            bos_sep_mass = attn_row[:, bos_sep_positions].sum(dim=1)
            top_is_ans = top_pos == 10
            top_is_bos_sep = torch.isin(top_pos, bos_sep_positions)
            top_is_number = torch.isin(top_pos, number_positions)
            top_is_max_number = torch.zeros(batch, dtype=torch.bool, device=device)
            for slot, pos in enumerate(NUMBER_POSITIONS.tolist()):
                top_is_max_number |= (top_pos == pos) & is_max_slot[:, slot]
            top_is_nonmax_number = top_is_number & ~top_is_max_number

            for true_max in range(2, 10):
                row_mask = labels == true_max
                if not bool(row_mask.any()):
                    continue
                if head_idx == 0:
                    counts_by_max[true_max] += int(row_mask.sum())
                mass_sums["max number"][head_idx, true_max] += float(max_mass[row_mask].sum())
                mass_sums["nonmax number"][head_idx, true_max] += float(
                    nonmax_number_mass[row_mask].sum()
                )
                mass_sums["ANS self"][head_idx, true_max] += float(self_mass[row_mask].sum())
                mass_sums["BOS/SEP"][head_idx, true_max] += float(bos_sep_mass[row_mask].sum())
                top_category_counts[f"H{head_idx}"][true_max, 0] += int(
                    top_is_max_number[row_mask].sum()
                )
                top_category_counts[f"H{head_idx}"][true_max, 1] += int(
                    top_is_nonmax_number[row_mask].sum()
                )
                top_category_counts[f"H{head_idx}"][true_max, 2] += int(top_is_ans[row_mask].sum())
                top_category_counts[f"H{head_idx}"][true_max, 3] += int(
                    top_is_bos_sep[row_mask].sum()
                )

        def final_digit_logits(head_values: list[torch.Tensor]) -> torch.Tensor:
            return digit_logits(ans_resid + layer.W_O(torch.cat(head_values, dim=-1)), model)

        conditions = {
            "actual": actual_values,
            "H3 max one-hot": actual_values[:3] + [max_values[3]],
            "H0/H2 top + H3 max one-hot": [
                top_values[0],
                actual_values[1],
                top_values[2],
                max_values[3],
            ],
            "all heads top one-hot": top_values,
            "H0/H2/H3 max one-hot": [
                max_values[0],
                actual_values[1],
                max_values[2],
                max_values[3],
            ],
            "all heads max one-hot": max_values,
            "only H3 max one-hot": [
                zero_values[0],
                actual_values[1],
                zero_values[2],
                max_values[3],
            ],
        }

        for name, head_values in conditions.items():
            add_accuracy(
                counts,
                per_max_correct,
                per_max_total,
                name,
                final_digit_logits(head_values),
                labels,
            )

    avg_mass_percent = {}
    for category, tensor in mass_sums.items():
        avg_mass_percent[category] = {
            f"H{head}": {
                str(true_max): float(100.0 * tensor[head, true_max] / counts_by_max[true_max])
                for true_max in range(2, 10)
            }
            for head in range(4)
        }

    accuracy = {
        name: {
            "accuracy": counts[name]["correct"] / counts[name]["total"],
            "correct": counts[name]["correct"],
            "total": counts[name]["total"],
        }
        for name in condition_names
    }
    accuracy_by_true_max = {
        name: {
            str(true_max): float(per_max_correct[name][true_max] / per_max_total[true_max])
            for true_max in range(2, 10)
        }
        for name in condition_names
    }

    summary = {
        "n_inputs_all_numbers_ge_2": total,
        "condition_accuracy": accuracy,
        "condition_accuracy_by_true_max": accuracy_by_true_max,
        "counts_by_true_max": {str(i): int(counts_by_max[i]) for i in range(2, 10)},
        "avg_percent_attention_mass": avg_mass_percent,
        "top_key_category_rate_by_true_max": summarize_top_categories(
            top_category_counts, counts_by_max
        ),
        "condition_definitions": {
            "H3 max one-hot": "Replace only H3's ANS row by one-hot to a max-token position.",
            "H0/H2 top + H3 max one-hot": (
                "Replace H0 and H2 by one-hot to their actual top key; replace H3 by "
                "one-hot to a max-token position; keep H1 actual."
            ),
            "all heads top one-hot": "Replace each head by one-hot to that head's actual top key.",
            "H0/H2/H3 max one-hot": (
                "Replace H0, H2, and H3 by one-hot to a max-token position; keep H1 actual."
            ),
            "all heads max one-hot": "Force every head to one-hot attend to a max-token position.",
            "only H3 max one-hot": (
                "Ablate H0 and H2, keep H1 actual, and replace H3 by one-hot to a "
                "max-token position."
            ),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    true_max_values = list(range(2, 10))
    category_order = ["max number", "nonmax number", "ANS self", "BOS/SEP"]
    category_colors = {
        "max number": "#2563eb",
        "nonmax number": "#f97316",
        "ANS self": "#14b8a6",
        "BOS/SEP": "#94a3b8",
    }

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    head_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    acc_ax = fig.add_subplot(gs[2, :])

    for head, ax in enumerate(head_axes):
        bottom = torch.zeros(len(true_max_values))
        for category in category_order:
            values = torch.tensor(
                [
                    summary["avg_percent_attention_mass"][category][f"H{head}"][str(true_max)]
                    for true_max in true_max_values
                ]
            )
            ax.bar(
                true_max_values,
                values,
                bottom=bottom,
                label=category,
                color=category_colors[category],
                alpha=0.88,
            )
            bottom += values
        ax.set_title(f"Head {head}: ANS-row attention destination")
        ax.set_xticks(true_max_values)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% attention mass")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    acc_values = [accuracy[name]["accuracy"] for name in condition_names]
    acc_colors = ["#111827", "#7c3aed", "#2563eb", "#0f766e", "#f97316", "#ef4444", "#64748b"]
    acc_ax.bar(range(len(condition_names)), acc_values, color=acc_colors, alpha=0.88)
    acc_ax.set_xticks(range(len(condition_names)))
    acc_ax.set_xticklabels(condition_names, rotation=20, ha="right")
    acc_ax.set_ylim(0, 1.05)
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Final accuracy after one-hot ANS-row interventions")
    acc_ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(acc_values):
        acc_ax.text(idx, min(value + 0.025, 1.02), f"{value:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        "Model 1: what H0/H2 read, and which one-hot replacements preserve accuracy",
        fontsize=15,
    )
    fig.savefig(OUT, dpi=180)

    print("condition,accuracy,correct,total")
    for name in condition_names:
        row = accuracy[name]
        print(f"{name},{row['accuracy']:.6f},{row['correct']},{row['total']}")
    print("avg_attention_percent_by_true_max")
    print("head,true_max,max_number,nonmax_number,ans_self,bos_sep")
    for head in range(4):
        for true_max in true_max_values:
            values = [
                summary["avg_percent_attention_mass"][category][f"H{head}"][str(true_max)]
                for category in category_order
            ]
            print(
                f"{head},{true_max},"
                + ",".join(f"{value:.6f}" for value in values)
            )
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
