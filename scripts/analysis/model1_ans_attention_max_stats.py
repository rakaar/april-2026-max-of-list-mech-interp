#!/usr/bin/env python3
"""Measure whether Model 1 ANS attention attends to max-valued number positions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_ans_attention_max_stats.png"
STATS_OUT = ROOT / "docs" / "assets" / "model1_ans_attention_max_stats.json"
NUMBER_POSITIONS = torch.tensor([1, 3, 5, 7, 9])


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


def all_inputs(device: str):
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device)
    nums, tokens = all_inputs(device)
    number_positions = NUMBER_POSITIONS.to(device)

    max_vals = nums.max(dim=1).values
    is_max = nums == max_vals[:, None]
    n_examples = nums.shape[0]

    stats = [
        {
            "head": h,
            "top_number_is_max": 0,
            "top_all_is_max_number": 0,
            "single_max_top_number_is_max": 0,
            "single_max_count": 0,
            "max_position_attention_mass": 0.0,
            "number_position_attention_mass": 0.0,
            "ans_self_attention_mass": 0.0,
            "bos_sep_attention_mass": 0.0,
            "top_all_key_position_counts": [0] * 11,
            "top_number_slot_counts": [0] * 5,
            "top_number_value_counts": [0] * 10,
        }
        for h in range(4)
    ]

    batch_size = 4096
    with torch.no_grad():
        for start in range(0, n_examples, batch_size):
            end = min(start + batch_size, n_examples)
            batch_nums = nums[start:end]
            batch_is_max = is_max[start:end]
            batch_single_max = batch_is_max.sum(dim=1) == 1

            _, attention_patterns = model(tokens[start:end])
            ans_attn = attention_patterns[0][:, :, 10, :]
            number_attn = ans_attn[:, :, number_positions]

            batch_idx = torch.arange(end - start, device=device)
            for head in range(4):
                full = ans_attn[:, head]
                numbers_only = number_attn[:, head]
                top_all = full.argmax(dim=1)
                top_number_slot = numbers_only.argmax(dim=1)
                top_number_is_max = batch_is_max[batch_idx, top_number_slot]

                top_all_is_max_number = torch.zeros(end - start, dtype=torch.bool, device=device)
                for slot, pos in enumerate(NUMBER_POSITIONS.tolist()):
                    top_all_is_max_number |= (top_all == pos) & batch_is_max[:, slot]

                max_mass = (numbers_only * batch_is_max.float()).sum(dim=1)
                top_number_values = batch_nums[batch_idx, top_number_slot]

                stat = stats[head]
                stat["top_number_is_max"] += int(top_number_is_max.sum())
                stat["top_all_is_max_number"] += int(top_all_is_max_number.sum())
                stat["single_max_top_number_is_max"] += int((top_number_is_max & batch_single_max).sum())
                stat["single_max_count"] += int(batch_single_max.sum())
                stat["max_position_attention_mass"] += float(max_mass.sum())
                stat["number_position_attention_mass"] += float(numbers_only.sum(dim=1).sum())
                stat["ans_self_attention_mass"] += float(full[:, 10].sum())
                stat["bos_sep_attention_mass"] += float(full[:, [0, 2, 4, 6, 8]].sum())
                stat["top_all_key_position_counts"] = (
                    torch.tensor(stat["top_all_key_position_counts"])
                    + torch.bincount(top_all.cpu(), minlength=11)
                ).tolist()
                stat["top_number_slot_counts"] = (
                    torch.tensor(stat["top_number_slot_counts"])
                    + torch.bincount(top_number_slot.cpu(), minlength=5)
                ).tolist()
                stat["top_number_value_counts"] = (
                    torch.tensor(stat["top_number_value_counts"])
                    + torch.bincount(top_number_values.cpu(), minlength=10)
                ).tolist()

    for stat in stats:
        stat["n_examples"] = n_examples
        stat["top_number_is_max_rate"] = stat["top_number_is_max"] / n_examples
        stat["top_all_is_max_number_rate"] = stat["top_all_is_max_number"] / n_examples
        stat["single_max_top_number_is_max_rate"] = (
            stat["single_max_top_number_is_max"] / stat["single_max_count"]
        )
        stat["avg_max_position_attention_mass"] = stat["max_position_attention_mass"] / n_examples
        stat["avg_number_position_attention_mass"] = stat["number_position_attention_mass"] / n_examples
        stat["avg_ans_self_attention_mass"] = stat["ans_self_attention_mass"] / n_examples
        stat["avg_bos_sep_attention_mass"] = stat["bos_sep_attention_mass"] / n_examples

    OUT.parent.mkdir(parents=True, exist_ok=True)
    STATS_OUT.write_text(json.dumps(stats, indent=2) + "\n")

    heads = [s["head"] for s in stats]
    series = {
        "top number slot is max": [s["top_number_is_max_rate"] for s in stats],
        "top overall key is max": [s["top_all_is_max_number_rate"] for s in stats],
        "attn mass on max slots": [s["avg_max_position_attention_mass"] for s in stats],
        "attn mass on ANS self": [s["avg_ans_self_attention_mass"] for s in stats],
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    colors = ["#2563eb", "#0f766e", "#f97316", "#7c3aed"]
    for (label, values), offset, color in zip(series.items(), offsets, colors):
        xs = [h + offset for h in heads]
        ax.bar(xs, values, width=width, label=label, color=color)
    ax.set_xticks(heads)
    ax.set_xticklabels([f"H{h}" for h in heads])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction / average attention mass")
    ax.set_title("Model 1 ANS-row attention to max-valued number positions")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180)

    print("head,top_number_is_max,top_all_is_max_number,avg_max_mass,avg_number_mass,avg_self_mass")
    for stat in stats:
        print(
            f"{stat['head']},"
            f"{stat['top_number_is_max_rate']:.6f},"
            f"{stat['top_all_is_max_number_rate']:.6f},"
            f"{stat['avg_max_position_attention_mass']:.6f},"
            f"{stat['avg_number_position_attention_mass']:.6f},"
            f"{stat['avg_ans_self_attention_mass']:.6f}"
        )
    print(f"wrote,{OUT}")
    print(f"wrote,{STATS_OUT}")


if __name__ == "__main__":
    main()

