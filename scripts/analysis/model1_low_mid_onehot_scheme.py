#!/usr/bin/env python3
"""Test the one-hot scheme H0/H1/H2 -> ANS self, H3 -> max for max 2..6."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
JSON_OUT = ROOT / "docs" / "assets" / "model1_low_mid_onehot_scheme.json"
NUMBER_POSITIONS = torch.tensor([1, 3, 5, 7, 9])
BATCH_SIZE = 1024


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
    nums = torch.cartesian_prod(*[torch.arange(2, 7) for _ in range(5)]).to(device)
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device)
    nums, tokens = make_inputs(device)
    labels = nums.max(dim=1).values
    number_positions = NUMBER_POSITIONS.to(device)

    total = tokens.shape[0]
    actual_correct = 0
    onehot_correct = 0
    per_max_total = torch.zeros(10, dtype=torch.long)
    per_max_actual_correct = torch.zeros(10, dtype=torch.long)
    per_max_onehot_correct = torch.zeros(10, dtype=torch.long)
    top_key_matches = {
        "H0_top_is_ANS": 0,
        "H1_top_is_ANS": 0,
        "H2_top_is_ANS": 0,
        "H3_top_is_max": 0,
    }
    avg_attention_mass = {
        "H0_ANS": 0.0,
        "H1_ANS": 0.0,
        "H2_ANS": 0.0,
        "H3_max": 0.0,
    }
    max_abs_logit_diff = 0.0

    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            x = tokens[start:end]
            n = nums[start:end]
            y = labels[start:end]
            batch = end - start
            batch_idx = torch.arange(batch, device=device)

            logits, attention_patterns = model(x)
            actual_digit_logits = logits[:, 10, :10]
            actual_pred = actual_digit_logits.argmax(dim=1)
            actual_correct += int((actual_pred == y).sum())

            seq_len = x.shape[1]
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            resid = model.tok_embed(x) + model.pos_embed(positions)
            ans_resid = resid[:, 10, :]
            layer = model.layers[0]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
            is_max_slot = n == y[:, None]

            onehot_head_values = []
            for head_idx, head in enumerate(layer.heads):
                _, attn = head(resid, mask)
                attn_row = attn[:, 10, :]
                source_values = resid @ head.W_V.weight.detach().T

                if head_idx in (0, 1, 2):
                    onehot_head_values.append(source_values[:, 10, :])
                    key = f"H{head_idx}_top_is_ANS"
                    top_key_matches[key] += int((attn_row.argmax(dim=1) == 10).sum())
                    avg_attention_mass[f"H{head_idx}_ANS"] += float(attn_row[:, 10].sum())
                else:
                    number_attn = attn_row[:, number_positions]
                    max_attn = number_attn.masked_fill(~is_max_slot, -1.0)
                    max_slot = max_attn.argmax(dim=1)
                    max_pos = number_positions[max_slot]
                    onehot_head_values.append(source_values[batch_idx, max_pos])

                    top_key = attn_row.argmax(dim=1)
                    top_is_max = torch.zeros(batch, dtype=torch.bool, device=device)
                    for slot, pos in enumerate(NUMBER_POSITIONS.tolist()):
                        top_is_max |= (top_key == pos) & is_max_slot[:, slot]
                    top_key_matches["H3_top_is_max"] += int(top_is_max.sum())
                    avg_attention_mass["H3_max"] += float(
                        (number_attn * is_max_slot.float()).sum(dim=1).sum()
                    )

            onehot_digit_logits = digit_logits(
                ans_resid + layer.W_O(torch.cat(onehot_head_values, dim=-1)),
                model,
            )
            onehot_pred = onehot_digit_logits.argmax(dim=1)
            onehot_correct += int((onehot_pred == y).sum())
            max_abs_logit_diff = max(
                max_abs_logit_diff,
                float((actual_digit_logits - onehot_digit_logits).abs().max()),
            )

            for true_max in range(2, 7):
                mask_true = y == true_max
                per_max_total[true_max] += int(mask_true.sum())
                per_max_actual_correct[true_max] += int((actual_pred[mask_true] == y[mask_true]).sum())
                per_max_onehot_correct[true_max] += int((onehot_pred[mask_true] == y[mask_true]).sum())

    summary = {
        "scope": "all 5-number inputs with every number in 2..6, equivalently true max in 2..6",
        "n_inputs": total,
        "actual_accuracy": {
            "accuracy": actual_correct / total,
            "correct": actual_correct,
            "total": total,
        },
        "onehot_scheme": "H0/H1/H2 one-hot to ANS self; H3 one-hot to a max-valued number position",
        "onehot_accuracy": {
            "accuracy": onehot_correct / total,
            "correct": onehot_correct,
            "total": total,
        },
        "max_abs_digit_logit_diff_actual_vs_onehot": max_abs_logit_diff,
        "top_key_rates": {
            key: value / total for key, value in top_key_matches.items()
        },
        "avg_attention_mass": {
            key: value / total for key, value in avg_attention_mass.items()
        },
        "accuracy_by_true_max": {
            str(true_max): {
                "count": int(per_max_total[true_max]),
                "actual_accuracy": float(per_max_actual_correct[true_max] / per_max_total[true_max]),
                "onehot_accuracy": float(per_max_onehot_correct[true_max] / per_max_total[true_max]),
            }
            for true_max in range(2, 7)
        },
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"n_inputs,{total}")
    print(f"actual_accuracy,{actual_correct / total:.6f},{actual_correct}/{total}")
    print(f"onehot_accuracy,{onehot_correct / total:.6f},{onehot_correct}/{total}")
    print(f"max_abs_digit_logit_diff_actual_vs_onehot,{max_abs_logit_diff:.6f}")
    print("top_key_rates")
    for key, value in summary["top_key_rates"].items():
        print(f"{key},{value:.6f}")
    print("avg_attention_mass")
    for key, value in summary["avg_attention_mass"].items():
        print(f"{key},{value:.6f}")
    print("accuracy_by_true_max")
    for true_max, row in summary["accuracy_by_true_max"].items():
        print(f"{true_max},{row['count']},{row['actual_accuracy']:.6f},{row['onehot_accuracy']:.6f}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
