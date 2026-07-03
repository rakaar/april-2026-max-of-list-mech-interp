#!/usr/bin/env python3
"""Inspect attention on all Model 1 inputs containing only 0/1 numbers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_low01_attention.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_low01_attention.json"
NUMBER_POSITIONS = [1, 3, 5, 7, 9]


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


def main() -> None:
    model = load_model()
    nums = torch.cartesian_prod(*[torch.arange(2) for _ in range(5)])
    tokens = torch.empty((nums.shape[0], 11), dtype=torch.long)
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

    with torch.no_grad():
        logits, attention_patterns = model(tokens)

    ans_attn = attention_patterns[0][:, :, 10, :]
    preds = logits[:, -1, :10].argmax(dim=-1)
    labels = nums.max(dim=1).values
    is_max = nums == labels[:, None]

    summaries = []
    for head in range(4):
        attn = ans_attn[:, head]
        max_mass = (attn[:, NUMBER_POSITIONS] * is_max.float()).sum(dim=1)
        summaries.append(
            {
                "head": head,
                "top_position_counts": torch.bincount(attn.argmax(dim=1), minlength=11).tolist(),
                "avg_ans_self_mass": float(attn[:, 10].mean()),
                "avg_max_position_mass": float(max_mass.mean()),
                "top_overall_is_max_count": int(
                    sum(
                        any(int(attn[i].argmax()) == pos and bool(is_max[i, slot]) for slot, pos in enumerate(NUMBER_POSITIONS))
                        for i in range(nums.shape[0])
                    )
                ),
            }
        )

    data = {
        "inputs": nums.tolist(),
        "predictions": preds.tolist(),
        "labels": labels.tolist(),
        "all_correct": bool((preds == labels).all()),
        "summaries": summaries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")

    heads = [s["head"] for s in summaries]
    self_mass = [s["avg_ans_self_mass"] for s in summaries]
    max_mass = [s["avg_max_position_mass"] for s in summaries]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.35
    ax.bar([h - width / 2 for h in heads], self_mass, width=width, label="avg ANS self attention")
    ax.bar([h + width / 2 for h in heads], max_mass, width=width, label="avg max-position attention")
    ax.set_xticks(heads)
    ax.set_xticklabels([f"H{h}" for h in heads])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average attention mass over all 0/1 inputs")
    ax.set_title("Model 1 ANS-row attention on all inputs with only 0 and 1")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180)

    print(f"all_correct,{bool((preds == labels).all())}")
    print("head,avg_self,avg_max_mass,top_overall_is_max_count,top_position_counts")
    for item in summaries:
        print(
            f"{item['head']},{item['avg_ans_self_mass']:.6f},"
            f"{item['avg_max_position_mass']:.6f},"
            f"{item['top_overall_is_max_count']},{item['top_position_counts']}"
        )
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()

