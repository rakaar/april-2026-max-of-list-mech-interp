#!/usr/bin/env python3
"""Compare actual H3 attention to one-hot max attention on a concrete example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_onehot_h3_example.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_onehot_h3_example.json"
NUMS = [6, 8, 4, 7, 5]


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
    tokens = torch.tensor([tokenize(NUMS)], dtype=torch.long)
    max_value = max(NUMS)
    max_token_position = 1 + 2 * NUMS.index(max_value)

    with torch.no_grad():
        logits, attention_patterns = model(tokens)

    seq_len = tokens.shape[1]
    positions = torch.arange(seq_len).unsqueeze(0)
    resid = model.tok_embed(tokens).detach() + model.pos_embed(positions).detach()
    layer = model.layers[0]
    w_o = layer.W_O.weight.detach()

    head_outs = []
    h3_actual_value = None
    h3_onehot_value = None
    h3_actual_out = None
    h3_onehot_out = None
    h3_actual_logits = None
    h3_onehot_logits = None
    ans_attention_rows = []

    for head_idx, head in enumerate(layer.heads):
        d_head = head.d_head
        w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
        values = resid[0] @ head.W_V.weight.detach().T
        attn_row = attention_patterns[0][0, head_idx, 10].detach()
        ans_attention_rows.append(attn_row)
        value = attn_row @ values
        out = value @ w_o_head.T
        head_outs.append(out)

        if head_idx == 3:
            h3_actual_value = value
            h3_onehot_value = values[max_token_position]
            h3_actual_out = out
            h3_onehot_out = h3_onehot_value @ w_o_head.T
            h3_actual_logits = number_logits(h3_actual_out, model)
            h3_onehot_logits = number_logits(h3_onehot_out, model)

    ans_resid = resid[0, 10]
    actual_final = ans_resid + sum(head_outs)
    replace_h3_final = ans_resid + sum(head_outs[:3]) + h3_onehot_out
    only_h3_onehot_final = ans_resid + h3_onehot_out
    resid_only = ans_resid

    final_logit_sets = {
        "actual all heads": number_logits(actual_final, model),
        "replace H3 with one-hot max": number_logits(replace_h3_final, model),
        "resid + only H3 one-hot": number_logits(only_h3_onehot_final, model),
        "resid only": number_logits(resid_only, model),
    }

    h3_cosine = float(F.cosine_similarity(h3_actual_value, h3_onehot_value, dim=0))

    data = {
        "nums": NUMS,
        "tokens": tokens.tolist()[0],
        "max_value": max_value,
        "max_token_position": max_token_position,
        "prediction": int(logits[0, -1, :10].argmax()),
        "h3_actual_attention_to_max_position": float(ans_attention_rows[3][max_token_position]),
        "h3_actual_value_norm": float(h3_actual_value.norm()),
        "h3_onehot_value_norm": float(h3_onehot_value.norm()),
        "h3_actual_vs_onehot_value_cosine": h3_cosine,
        "h3_actual_ov_logits": [float(x) for x in h3_actual_logits],
        "h3_onehot_ov_logits": [float(x) for x in h3_onehot_logits],
        "final_logit_sets": {name: [float(x) for x in values] for name, values in final_logit_sets.items()},
        "ans_attention_rows": {
            f"H{idx}": [float(x) for x in row] for idx, row in enumerate(ans_attention_rows)
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    labels = [f"{tok}@{pos}" for pos, tok in enumerate(tokens.tolist()[0])]

    ax = axes[0, 0]
    for head_idx, row in enumerate(ans_attention_rows):
        ax.plot(range(seq_len), row.numpy(), marker="o", label=f"H{head_idx}")
    ax.axvline(max_token_position, color="#dc2626", linestyle="--", label="max token position")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Attention probability")
    ax.set_title(f"ANS attention rows for {NUMS}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    xs = range(10)
    ax.plot(xs, h3_actual_logits.numpy(), marker="o", label="H3 actual attention")
    ax.plot(xs, h3_onehot_logits.numpy(), marker="o", linestyle="--", label="H3 one-hot to max")
    ax.set_xticks(xs)
    ax.set_xlabel("Output number logit")
    ax.set_ylabel("H3 OV logit contribution")
    ax.set_title(f"H3 actual vs one-hot value, cosine={h3_cosine:.6f}")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    names = list(final_logit_sets.keys())
    for name in names:
        ax.plot(xs, final_logit_sets[name].numpy(), marker="o", label=name)
    ax.axvline(max_value, color="#dc2626", linestyle="--", label="true max")
    ax.set_xticks(xs)
    ax.set_xlabel("Output number logit")
    ax.set_ylabel("Final/residual logit")
    ax.set_title("Final logits under H3 one-hot intervention")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    top_rows = []
    row_labels = []
    for name, values in final_logit_sets.items():
        top_rows.append(values.numpy())
        row_labels.append(f"{name}\npred={int(values.argmax())}")
    im = ax.imshow(top_rows, cmap="coolwarm", aspect="auto")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(xs)
    ax.set_xlabel("Output number logit")
    ax.set_title("Number logits by condition")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Model 1: what H3 writes when it attends to the max token", fontsize=15)
    fig.savefig(OUT, dpi=180)

    print(f"nums,{NUMS},max,{max_value},max_token_position,{max_token_position}")
    print(f"prediction,{data['prediction']}")
    print(f"h3_attention_to_max,{data['h3_actual_attention_to_max_position']:.6f}")
    print(f"h3_actual_vs_onehot_value_cosine,{h3_cosine:.6f}")
    for name, values in final_logit_sets.items():
        print(f"{name},pred={int(values.argmax())}," + ",".join(f"{i}:{float(values[i]):+.3f}" for i in range(10)))
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()

