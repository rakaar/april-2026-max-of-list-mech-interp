#!/usr/bin/env python3
"""Decompose the logit[7] - logit[6] margin on selected examples."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_margin_7v6_examples.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_margin_7v6_examples.json"

EXAMPLES = [
    [6, 2, 3, 4, 5],
    [2, 6, 4, 5, 3],
    [2, 3, 6, 4, 5],
    [5, 4, 3, 2, 6],
    [6, 6, 5, 4, 3],
    [7, 2, 3, 4, 5],
    [2, 7, 4, 6, 3],
    [2, 3, 7, 4, 6],
    [6, 5, 4, 3, 7],
    [7, 7, 6, 5, 4],
]

NUMBER_POSITIONS = [1, 3, 5, 7, 9]
COMPONENTS = ["residual", "H0", "H1", "H2", "H3", "final"]


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


def main() -> None:
    model = load_model()
    tokens = torch.tensor([tokenize(nums) for nums in EXAMPLES], dtype=torch.long)
    nums_t = torch.tensor(EXAMPLES, dtype=torch.long)
    labels = nums_t.max(dim=1).values

    with torch.no_grad():
        model_logits, attention_patterns = model(tokens)
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
            head_vec = head_values[:, 10, :] @ w_o_head.T
            head_vectors.append(head_vec)
            head_attn_rows.append(attn[:, 10, :])

        component_vectors = {
            "residual": ans_resid,
            "H0": head_vectors[0],
            "H1": head_vectors[1],
            "H2": head_vectors[2],
            "H3": head_vectors[3],
        }
        final_vec = ans_resid + sum(head_vectors)
        component_vectors["final"] = final_vec

        component_logits = {
            name: number_logits(vec, model) for name, vec in component_vectors.items()
        }
        margins_7v6 = {
            name: logits[:, 7] - logits[:, 6] for name, logits in component_logits.items()
        }

    rows = []
    for idx, nums in enumerate(EXAMPLES):
        max_value = int(labels[idx])
        row = {
            "example_index": idx,
            "nums": nums,
            "true_max": max_value,
            "model_pred": int(model_logits[idx, 10, :10].argmax()),
            "component_logits_6_7": {
                name: {
                    "logit6": float(component_logits[name][idx, 6]),
                    "logit7": float(component_logits[name][idx, 7]),
                    "margin_7_minus_6": float(margins_7v6[name][idx]),
                }
                for name in COMPONENTS
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
    for true_max in [6, 7]:
        indices = [i for i, row in enumerate(rows) if row["true_max"] == true_max]
        summary_by_true_max[str(true_max)] = {
            name: float(torch.stack([margins_7v6[name][i] for i in indices]).mean())
            for name in COMPONENTS
        }

    result = {
        "description": "Component decomposition of logit[7] - logit[6] on selected examples.",
        "components": COMPONENTS,
        "examples": rows,
        "avg_margin_7_minus_6_by_true_max": summary_by_true_max,
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    margin_matrix = torch.stack([margins_7v6[name] for name in COMPONENTS])
    fig, ax = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
    vmax = max(abs(float(margin_matrix.min())), abs(float(margin_matrix.max())))
    im = ax.imshow(margin_matrix.detach().numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    labels_x = [f"{i}: {nums}\nmax={int(labels[i])}" for i, nums in enumerate(EXAMPLES)]
    ax.set_xticks(range(batch))
    ax.set_xticklabels(labels_x, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(COMPONENTS)))
    ax.set_yticklabels(COMPONENTS)
    ax.set_title("Component contributions to logit[7] - logit[6]")
    for y in range(len(COMPONENTS)):
        for x in range(batch):
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
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="logit[7] - logit[6]")
    fig.savefig(OUT, dpi=180)

    print("example,true_max,pred,component,logit6,logit7,margin_7_minus_6")
    for row in rows:
        for name in COMPONENTS:
            item = row["component_logits_6_7"][name]
            print(
                f"{row['example_index']},{row['true_max']},{row['model_pred']},"
                f"{name},{item['logit6']:+.6f},{item['logit7']:+.6f},"
                f"{item['margin_7_minus_6']:+.6f}"
            )
    print("avg_margin_7_minus_6_by_true_max")
    for true_max, values in summary_by_true_max.items():
        print(
            true_max + ","
            + ",".join(f"{name}:{values[name]:+.6f}" for name in COMPONENTS)
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
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
