#!/usr/bin/env python3
"""Decompose head 3 QK scores from ANS@pos10 to number keys."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_h3_ans_qk_decomposition.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_h3_ans_qk_decomposition.json"
NUMBER_POSITIONS = [1, 3, 5, 7, 9]
SPECIAL_POSITIONS = [(0, 10, "BOS"), (2, 11, "SEP"), (4, 11, "SEP"), (6, 11, "SEP"), (8, 11, "SEP"), (10, 12, "ANS")]


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


def score(q: torch.Tensor, k: torch.Tensor, d_head: int) -> torch.Tensor:
    if k.ndim == 1:
        return q @ k / (d_head**0.5)
    return q @ k.T / (d_head**0.5)


def main() -> None:
    model = load_model()
    head = model.layers[0].heads[3]
    embeddings = model.tok_embed.weight.detach()
    positions = model.pos_embed.weight.detach()
    w_q = head.W_Q.weight.detach()
    w_k = head.W_K.weight.detach()
    d_head = head.d_head

    ans_token = 12
    q_token = embeddings[ans_token] @ w_q.T
    q_pos = positions[10] @ w_q.T
    q_full = q_token + q_pos

    number_key_token = embeddings[:10] @ w_k.T
    number_key_pos = positions[NUMBER_POSITIONS] @ w_k.T

    token_token = score(q_token, number_key_token, d_head)
    pos_token = score(q_pos, number_key_token, d_head)
    token_dependent = token_token + pos_token

    token_pos = score(q_token, number_key_pos, d_head)
    pos_pos = score(q_pos, number_key_pos, d_head)
    position_bias = token_pos + pos_pos

    full_scores = token_dependent[None, :] + position_bias[:, None]
    special_scores = []
    for pos, tok, label in SPECIAL_POSITIONS:
        key = (embeddings[tok] + positions[pos]) @ w_k.T
        special_scores.append({"position": pos, "token": label, "score": float(score(q_full, key, d_head))})

    data = {
        "head": 3,
        "query": "ANS@10",
        "number_positions": NUMBER_POSITIONS,
        "token_token": token_token.tolist(),
        "pos_token": pos_token.tolist(),
        "token_dependent": token_dependent.tolist(),
        "token_pos": token_pos.tolist(),
        "pos_pos": pos_pos.tolist(),
        "position_bias": position_bias.tolist(),
        "full_scores": full_scores.tolist(),
        "special_scores": special_scores,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    xs = list(range(10))
    ax.plot(xs, token_token.numpy(), marker="o", label="E_ANS Q dot E_num K")
    ax.plot(xs, pos_token.numpy(), marker="o", label="P_10 Q dot E_num K")
    ax.plot(xs, token_dependent.numpy(), marker="o", linewidth=2.5, label="sum")
    ax.axhline(0, color="#d1d5db", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xlabel("Number key token")
    ax.set_ylabel("QK score contribution")
    ax.set_title("H3 token-dependent QK score from ANS@10")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    labels = [f"pos {p}" for p in NUMBER_POSITIONS]
    ax.bar(labels, position_bias.numpy(), color="#0f766e")
    ax.set_ylim(min(position_bias).item() - 0.02, max(position_bias).item() + 0.02)
    ax.set_ylabel("QK score contribution")
    ax.set_title("H3 position bias for number slots is almost flat")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    im = ax.imshow(full_scores.numpy(), cmap="coolwarm", aspect="auto")
    ax.set_xticks(xs)
    ax.set_yticks(range(len(NUMBER_POSITIONS)))
    ax.set_yticklabels([f"pos {p}" for p in NUMBER_POSITIONS])
    ax.set_xlabel("Number key token")
    ax.set_ylabel("Number key position")
    ax.set_title("H3 full ANS@10 QK score to number keys")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    sp_labels = [f"{s['token']}@{s['position']}" for s in special_scores]
    sp_scores = [s["score"] for s in special_scores]
    ax.bar(sp_labels, sp_scores, color="#7c3aed")
    ax.axhline(float(full_scores[:, 2].min()), color="#2563eb", linestyle="--", linewidth=1, label="min score for number 2")
    ax.axhline(float(full_scores[:, 1].max()), color="#dc2626", linestyle="--", linewidth=1, label="max score for number 1")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("QK score")
    ax.set_title("Special-token scores explain low-number self exceptions")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Model 1 H3: why ANS attends to the maximum number", fontsize=15)
    fig.savefig(OUT, dpi=180)

    print("number,token_token,pos_token,sum")
    for n in range(10):
        print(f"{n},{float(token_token[n]):+.6f},{float(pos_token[n]):+.6f},{float(token_dependent[n]):+.6f}")
    print("position,position_bias")
    for pos, bias in zip(NUMBER_POSITIONS, position_bias):
        print(f"{pos},{float(bias):+.6f}")
    print("special_token_scores")
    for item in special_scores:
        print(f"{item['token']}@{item['position']},{item['score']:+.6f}")
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
