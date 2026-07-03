#!/usr/bin/env python3
"""Plot ANS@pos10 query dot number-token keys for each Model 1 head."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_ans_query_number_key_scores.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_ans_query_number_key_scores.json"


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
    embeddings = model.tok_embed.weight.detach()
    positions = model.pos_embed.weight.detach()
    ans_resid = embeddings[12] + positions[10]
    number_embeddings = embeddings[:10]
    numbers = torch.arange(10, dtype=torch.float32)

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for head_idx, head in enumerate(model.layers[0].heads):
        w_q = head.W_Q.weight.detach()
        w_k = head.W_K.weight.detach()
        q = ans_resid @ w_q.T
        k_numbers = number_embeddings @ w_k.T
        scores = q @ k_numbers.T / (head.d_head**0.5)

        corr = torch.corrcoef(torch.stack([numbers, scores]))[0, 1]
        monotonic = bool(all(scores[i] <= scores[i + 1] for i in range(9)))

        rows.append(
            {
                "head": head_idx,
                "scores": [float(x) for x in scores],
                "corr_with_number": float(corr),
                "monotonic_increasing": monotonic,
            }
        )

        axes[0].plot(range(10), scores.numpy(), marker="o", label=f"H{head_idx}")
        axes[1].bar(
            [x + (head_idx - 1.5) * 0.18 for x in range(10)],
            scores.numpy(),
            width=0.18,
            label=f"H{head_idx}",
        )

    axes[0].axhline(0, color="#d1d5db", linewidth=0.8)
    axes[0].set_xticks(range(10))
    axes[0].set_xlabel("Number key token")
    axes[0].set_ylabel("Scaled QK score")
    axes[0].set_title("(E[ANS] + P[10]) W_Q · E[number] W_K")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].axhline(0, color="#d1d5db", linewidth=0.8)
    axes[1].set_xticks(range(10))
    axes[1].set_xlabel("Number key token")
    axes[1].set_ylabel("Scaled QK score")
    axes[1].set_title("Grouped by number token")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Model 1 ANS@pos10 query score to number-token keys", fontsize=14)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    JSON_OUT.write_text(json.dumps(rows, indent=2) + "\n")

    print("head,number,score")
    for row in rows:
        for number, score in enumerate(row["scores"]):
            print(f"{row['head']},{number},{score:+.6f}")
        print(
            f"summary_head_{row['head']},corr={row['corr_with_number']:+.6f},"
            f"monotonic_increasing={row['monotonic_increasing']}"
        )
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()

