#!/usr/bin/env python3
"""Compare H3 ANS@10 self-key QK score to number-key QK scores."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_h3_ans_self_vs_numbers.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_h3_ans_self_vs_numbers.json"
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


def scaled_dot(q: torch.Tensor, k: torch.Tensor, d_head: int) -> float:
    return float(q @ k / (d_head**0.5))


def main() -> None:
    model = load_model()
    head = model.layers[0].heads[3]
    embeddings = model.tok_embed.weight.detach()
    positions = model.pos_embed.weight.detach()
    w_q = head.W_Q.weight.detach()
    w_k = head.W_K.weight.detach()

    q = (embeddings[12] + positions[10]) @ w_q.T
    self_key = (embeddings[12] + positions[10]) @ w_k.T
    self_score = scaled_dot(q, self_key, head.d_head)

    token_only_scores = []
    full_by_position = []
    for n in range(10):
        token_only_scores.append(scaled_dot(q, embeddings[n] @ w_k.T, head.d_head))

    for pos in NUMBER_POSITIONS:
        row = []
        for n in range(10):
            row.append(scaled_dot(q, (embeddings[n] + positions[pos]) @ w_k.T, head.d_head))
        full_by_position.append(row)

    full_tensor = torch.tensor(full_by_position)
    full_min = full_tensor.min(dim=0).values
    full_max = full_tensor.max(dim=0).values
    full_mean = full_tensor.mean(dim=0)

    data = {
        "head": 3,
        "query": "ANS@10",
        "self_score": self_score,
        "token_only_number_scores": token_only_scores,
        "number_positions": NUMBER_POSITIONS,
        "full_number_scores_by_position": full_by_position,
        "self_beats_token_only_numbers": [
            n for n, score in enumerate(token_only_scores) if self_score > score
        ],
        "self_beats_full_number_keys": [
            n for n in range(10) if all(self_score > full_by_position[i][n] for i in range(5))
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")

    xs = list(range(10))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(xs, token_only_scores, marker="o", label="number token only: E[n]K")
    ax.plot(xs, full_mean.numpy(), marker="o", label="number token + position: mean over slots")
    ax.fill_between(
        xs,
        full_min.numpy(),
        full_max.numpy(),
        color="#93c5fd",
        alpha=0.25,
        label="number token + position: min/max over slots",
    )
    ax.axhline(self_score, color="#dc2626", linestyle="--", linewidth=1.5, label="ANS@10 self key")
    ax.set_xticks(xs)
    ax.set_xlabel("Number key token")
    ax.set_ylabel("Scaled QK score")
    ax.set_title("Model 1 H3: ANS@10 self score vs number-key scores")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT, dpi=180)

    print(f"self_score,{self_score:+.6f}")
    print(
        "token_only,"
        + ",".join(f"{n}:{score:+.6f}" for n, score in enumerate(token_only_scores))
    )
    print(
        "full_mean,"
        + ",".join(f"{n}:{float(score):+.6f}" for n, score in enumerate(full_mean))
    )
    print(f"self_beats_token_only_numbers,{data['self_beats_token_only_numbers']}")
    print(f"self_beats_full_number_keys,{data['self_beats_full_number_keys']}")
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()

