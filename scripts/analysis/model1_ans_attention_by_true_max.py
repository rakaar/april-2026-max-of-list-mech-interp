#!/usr/bin/env python3
"""Group ANS-row attention-to-max statistics by true maximum value."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_ans_attention_by_true_max.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_ans_attention_by_true_max.json"
NUMBER_POSITIONS = torch.tensor([1, 3, 5, 7, 9])
BATCH_SIZE = 4096


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
    nums, tokens = make_inputs(device)
    true_max = nums.max(dim=1).values
    number_positions = NUMBER_POSITIONS.to(device)

    counts_by_max = torch.zeros(10, dtype=torch.long)
    max_mass_sum = torch.zeros(4, 10)
    number_mass_sum = torch.zeros(4, 10)
    self_mass_sum = torch.zeros(4, 10)
    other_mass_sum = torch.zeros(4, 10)
    top_key_is_max_count = torch.zeros(4, 10, dtype=torch.long)

    with torch.no_grad():
        for start in range(0, tokens.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, tokens.shape[0])
            batch_nums = nums[start:end]
            batch_true_max = true_max[start:end]
            batch_is_max = batch_nums == batch_true_max[:, None]

            _, attention_patterns = model(tokens[start:end])
            ans_attn = attention_patterns[0][:, :, 10, :]
            number_attn = ans_attn[:, :, number_positions]

            top_all = ans_attn.argmax(dim=2)
            top_key_is_max = torch.zeros((end - start, 4), dtype=torch.bool, device=device)
            for slot_idx, pos in enumerate(NUMBER_POSITIONS.tolist()):
                top_key_is_max |= (top_all == pos) & batch_is_max[:, slot_idx].unsqueeze(1)

            max_mass = (number_attn * batch_is_max[:, None, :].float()).sum(dim=2)
            number_mass = number_attn.sum(dim=2)
            self_mass = ans_attn[:, :, 10]
            other_mass = 1.0 - max_mass

            for max_value in range(10):
                mask = batch_true_max == max_value
                if not bool(mask.any()):
                    continue
                counts_by_max[max_value] += int(mask.sum())
                max_mass_sum[:, max_value] += max_mass[mask].sum(dim=0).cpu()
                number_mass_sum[:, max_value] += number_mass[mask].sum(dim=0).cpu()
                self_mass_sum[:, max_value] += self_mass[mask].sum(dim=0).cpu()
                other_mass_sum[:, max_value] += other_mass[mask].sum(dim=0).cpu()
                top_key_is_max_count[:, max_value] += top_key_is_max[mask].sum(dim=0).cpu()

    denom = counts_by_max.float().clamp_min(1.0)
    avg_max_mass = max_mass_sum / denom
    avg_number_mass = number_mass_sum / denom
    avg_self_mass = self_mass_sum / denom
    avg_other_mass = other_mass_sum / denom
    top_key_is_max_rate = top_key_is_max_count.float() / denom

    data = {
        "definition": (
            "For each input, max attention mass is the sum of ANS-row attention "
            "probability assigned to all number positions whose token equals the "
            "true maximum. If max repeats, all repeated max positions are included."
        ),
        "n_inputs_total": int(tokens.shape[0]),
        "counts_by_true_max": {str(i): int(counts_by_max[i]) for i in range(10)},
        "avg_percent_attention_to_max_tokens": {
            f"H{head}": {str(i): float(100.0 * avg_max_mass[head, i]) for i in range(10)}
            for head in range(4)
        },
        "avg_percent_attention_to_number_positions": {
            f"H{head}": {str(i): float(100.0 * avg_number_mass[head, i]) for i in range(10)}
            for head in range(4)
        },
        "avg_percent_attention_to_ans_self": {
            f"H{head}": {str(i): float(100.0 * avg_self_mass[head, i]) for i in range(10)}
            for head in range(4)
        },
        "top_overall_key_is_max_token_rate": {
            f"H{head}": {str(i): float(top_key_is_max_rate[head, i]) for i in range(10)}
            for head in range(4)
        },
        "avg_percent_attention_not_to_max_tokens": {
            f"H{head}": {str(i): float(100.0 * avg_other_mass[head, i]) for i in range(10)}
            for head in range(4)
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = ["#2563eb", "#14b8a6", "#f97316", "#7c3aed"]
    max_values = list(range(10))

    for head, ax in enumerate(axes):
        percents = [100.0 * float(avg_max_mass[head, value]) for value in max_values]
        ax.bar(max_values, percents, color=colors[head], alpha=0.86)
        ax.plot(
            max_values,
            [100.0 * float(top_key_is_max_rate[head, value]) for value in max_values],
            color="#111827",
            marker="o",
            linewidth=1.5,
            label="top key is max (%)",
        )
        ax.set_title(f"Head {head}")
        ax.set_xticks(max_values)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
        if head in (0, 2):
            ax.set_ylabel("% ANS attention to max-token positions")
        if head in (2, 3):
            ax.set_xlabel("True max value")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Model 1: ANS-row attention to max-valued number positions by true max",
        fontsize=15,
    )
    fig.savefig(OUT, dpi=180)

    print("true_max,count," + ",".join(f"H{h}_avg_pct_max_attn" for h in range(4)))
    for max_value in range(10):
        values = ",".join(f"{100.0 * float(avg_max_mass[h, max_value]):.6f}" for h in range(4))
        print(f"{max_value},{int(counts_by_max[max_value])},{values}")
    print("top_key_is_max_rate_by_true_max")
    print("true_max," + ",".join(f"H{h}" for h in range(4)))
    for max_value in range(10):
        values = ",".join(f"{float(top_key_is_max_rate[h, max_value]):.6f}" for h in range(4))
        print(f"{max_value},{values}")
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
