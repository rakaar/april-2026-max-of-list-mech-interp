#!/usr/bin/env python3
"""Test one-hot ANS-row schemes for true max 0 and 1."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
JSON_OUT = ROOT / "docs" / "assets" / "model1_low01_onehot_scheme.json"
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


def make_inputs(device: str):
    nums = torch.cartesian_prod(*[torch.arange(2) for _ in range(5)]).to(device)
    labels = nums.max(dim=1).values
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
    return nums, tokens, labels


def digit_logits(vec: torch.Tensor, model) -> torch.Tensor:
    return vec @ model.unembed.weight.detach()[:10].T


def add_accuracy(
    pred: torch.Tensor,
    labels: torch.Tensor,
    correct: dict[str, int],
    per_max_correct: dict[str, torch.Tensor],
    name: str,
) -> None:
    correct[name] += int((pred == labels).sum())
    for true_max in range(2):
        mask = labels == true_max
        per_max_correct[name][true_max] += int((pred[mask] == labels[mask]).sum())


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device)
    nums, tokens, labels = make_inputs(device)
    number_positions = NUMBER_POSITIONS.to(device)
    total = tokens.shape[0]

    condition_names = [
        "actual",
        "H0_H1_H2_ANS_onehot_H3_actual",
        "H3_top_onehot_others_actual",
        "all_heads_ans_onehot",
        "all_heads_top_onehot",
        "h3_max_others_ans_onehot",
        "all_heads_max_onehot",
    ]
    correct = {name: 0 for name in condition_names}
    per_max_correct = {name: torch.zeros(2, dtype=torch.long) for name in condition_names}
    per_max_total = torch.zeros(2, dtype=torch.long)
    top_key_counts = {
        f"H{head}_top_is_ANS": torch.zeros(2, dtype=torch.long) for head in range(4)
    }
    top_key_counts.update(
        {f"H{head}_top_is_max": torch.zeros(2, dtype=torch.long) for head in range(4)}
    )
    avg_attention_mass = {}
    for head in range(4):
        avg_attention_mass[f"H{head}_ANS"] = torch.zeros(2)
        avg_attention_mass[f"H{head}_max"] = torch.zeros(2)
    max_abs_logit_diff = {name: 0.0 for name in condition_names if name != "actual"}

    with torch.no_grad():
        logits, _ = model(tokens)
        actual_digit_logits = logits[:, 10, :10]
        actual_pred = actual_digit_logits.argmax(dim=1)
        add_accuracy(actual_pred, labels, correct, per_max_correct, "actual")

        for true_max in range(2):
            per_max_total[true_max] = int((labels == true_max).sum())

        seq_len = tokens.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        resid = model.tok_embed(tokens) + model.pos_embed(positions)
        ans_resid = resid[:, 10, :]
        layer = model.layers[0]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
        is_max_slot = nums == labels[:, None]
        batch_idx = torch.arange(total, device=device)

        self_values = []
        max_values = []
        top_values = []
        actual_values = []

        for head_idx, head in enumerate(layer.heads):
            out, attn = head(resid, mask)
            attn_row = attn[:, 10, :]
            source_values = resid @ head.W_V.weight.detach().T
            number_attn = attn_row[:, number_positions]

            max_attn = number_attn.masked_fill(~is_max_slot, -1.0)
            max_slot = max_attn.argmax(dim=1)
            max_pos = number_positions[max_slot]
            top_pos = attn_row.argmax(dim=1)

            self_values.append(source_values[:, 10, :])
            max_values.append(source_values[batch_idx, max_pos])
            top_values.append(source_values[batch_idx, top_pos])
            actual_values.append(out[:, 10, :])

            top_is_max = torch.zeros(total, dtype=torch.bool, device=device)
            for slot, pos in enumerate(NUMBER_POSITIONS.tolist()):
                top_is_max |= (top_pos == pos) & is_max_slot[:, slot]

            for true_max in range(2):
                row_mask = labels == true_max
                top_key_counts[f"H{head_idx}_top_is_ANS"][true_max] += int(
                    (top_pos[row_mask] == 10).sum()
                )
                top_key_counts[f"H{head_idx}_top_is_max"][true_max] += int(
                    top_is_max[row_mask].sum()
                )
                avg_attention_mass[f"H{head_idx}_ANS"][true_max] += float(
                    attn_row[row_mask, 10].sum()
                )
                avg_attention_mass[f"H{head_idx}_max"][true_max] += float(
                    (number_attn[row_mask] * is_max_slot[row_mask].float()).sum()
                )

        conditions = {
            "H0_H1_H2_ANS_onehot_H3_actual": [
                self_values[0],
                self_values[1],
                self_values[2],
                actual_values[3],
            ],
            "H3_top_onehot_others_actual": [
                actual_values[0],
                actual_values[1],
                actual_values[2],
                top_values[3],
            ],
            "all_heads_ans_onehot": self_values,
            "all_heads_top_onehot": top_values,
            "h3_max_others_ans_onehot": [
                self_values[0],
                self_values[1],
                self_values[2],
                max_values[3],
            ],
            "all_heads_max_onehot": max_values,
        }

        for name, head_values in conditions.items():
            intervention_logits = digit_logits(
                ans_resid + layer.W_O(torch.cat(head_values, dim=-1)),
                model,
            )
            pred = intervention_logits.argmax(dim=1)
            add_accuracy(pred, labels, correct, per_max_correct, name)
            max_abs_logit_diff[name] = float((actual_digit_logits - intervention_logits).abs().max())

    accuracy = {
        name: {
            "accuracy": correct[name] / total,
            "correct": correct[name],
            "total": total,
        }
        for name in condition_names
    }
    accuracy_by_true_max = {
        name: {
            str(true_max): {
                "accuracy": float(per_max_correct[name][true_max] / per_max_total[true_max]),
                "correct": int(per_max_correct[name][true_max]),
                "total": int(per_max_total[true_max]),
            }
            for true_max in range(2)
        }
        for name in condition_names
    }
    top_key_rates = {
        key: {
            str(true_max): float(value[true_max] / per_max_total[true_max])
            for true_max in range(2)
        }
        for key, value in top_key_counts.items()
    }
    avg_attention = {
        key: {
            str(true_max): float(value[true_max] / per_max_total[true_max])
            for true_max in range(2)
        }
        for key, value in avg_attention_mass.items()
    }

    summary = {
        "scope": "all inputs whose numbers are only 0 or 1",
        "n_inputs": total,
        "scheme": {
            "true_max_0": "H0 -> ANS, H1 -> ANS, H2 -> ANS, H3 -> ANS",
            "true_max_1": "H0 -> ANS, H1 -> ANS, H2 -> ANS, H3 -> ANS",
        },
        "condition_accuracy": accuracy,
        "condition_accuracy_by_true_max": accuracy_by_true_max,
        "top_key_rates": top_key_rates,
        "avg_attention_mass": avg_attention,
        "max_abs_digit_logit_diff_vs_actual": max_abs_logit_diff,
        "counts_by_true_max": {str(i): int(per_max_total[i]) for i in range(2)},
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"n_inputs,{total}")
    print("condition,accuracy,correct,total")
    for name in condition_names:
        row = accuracy[name]
        print(f"{name},{row['accuracy']:.6f},{row['correct']}/{row['total']}")
    print("accuracy_by_true_max")
    print("condition,true_max,accuracy,correct,total")
    for name in condition_names:
        for true_max in range(2):
            row = accuracy_by_true_max[name][str(true_max)]
            print(f"{name},{true_max},{row['accuracy']:.6f},{row['correct']}/{row['total']}")
    print("top_key_rates")
    for key, values in top_key_rates.items():
        print(key + "," + ",".join(f"{m}:{values[str(m)]:.6f}" for m in range(2)))
    print("avg_attention_mass")
    for key, values in avg_attention.items():
        print(key + "," + ",".join(f"{m}:{values[str(m)]:.6f}" for m in range(2)))
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
