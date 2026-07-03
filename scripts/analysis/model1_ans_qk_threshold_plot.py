#!/usr/bin/env python3
"""Plot per-head ANS query number-key QK scores against the ANS self threshold."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_ans_qk_threshold_plot.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_ans_qk_threshold_plot.json"


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

    results = []
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=False, constrained_layout=True)

    for head_idx, head in enumerate(model.layers[0].heads):
        w_q = head.W_Q.weight.detach()
        w_k = head.W_K.weight.detach()
        scale = head.d_head**0.5

        q_ans = ans_resid @ w_q.T
        k_numbers_token_only = number_embeddings @ w_k.T
        k_ans_self = ans_resid @ w_k.T

        number_scores = q_ans @ k_numbers_token_only.T / scale
        ans_self_score = q_ans @ k_ans_self / scale
        above_self = number_scores > ans_self_score

        recruited_numbers = [int(i) for i, is_above in enumerate(above_self) if bool(is_above)]
        threshold_label = "none" if not recruited_numbers else ",".join(map(str, recruited_numbers))
        results.append(
            {
                "head": head_idx,
                "query": "(E[ANS] + P[10]) @ W_Q.T",
                "number_keys": "E[number] @ W_K.T",
                "ans_self_key": "(E[ANS] + P[10]) @ W_K.T",
                "number_scores": [float(x) for x in number_scores],
                "ans_self_score": float(ans_self_score),
                "numbers_above_ans_self": recruited_numbers,
            }
        )

        ax = axes[head_idx]
        colors = ["#2563eb" if bool(is_above) else "#94a3b8" for is_above in above_self]
        ax.bar(range(10), number_scores.numpy(), color=colors, alpha=0.9)
        ax.axhline(
            float(ans_self_score),
            color="#dc2626",
            linestyle="--",
            linewidth=1.8,
            label=f"ANS self = {float(ans_self_score):.2f}",
        )
        ax.plot(range(10), number_scores.numpy(), color="#111827", marker="o", linewidth=1.2)
        ax.set_title(f"H{head_idx}: numbers above self = {threshold_label}")
        ax.set_xticks(range(10))
        ax.set_xlabel("Number key token")
        if head_idx == 0:
            ax.set_ylabel("Scaled QK score")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Model 1: ANS@pos10 query scores to number-token keys vs ANS self threshold",
        fontsize=15,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    JSON_OUT.write_text(json.dumps(results, indent=2) + "\n")

    print("head,ans_self_score,numbers_above_self")
    for row in results:
        above = "none" if not row["numbers_above_ans_self"] else "+".join(
            str(x) for x in row["numbers_above_ans_self"]
        )
        print(f"{row['head']},{row['ans_self_score']:.6f},{above}")
        for number, score in enumerate(row["number_scores"]):
            label = "above" if number in row["numbers_above_ans_self"] else "below"
            print(f"  {number},{score:.6f},{label}")
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
