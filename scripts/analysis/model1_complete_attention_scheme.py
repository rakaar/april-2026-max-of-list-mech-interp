#!/usr/bin/env python3
"""Verify the complete piecewise ANS-row attention abstraction over all inputs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
JSON_OUT = ROOT / "docs" / "assets" / "model1_complete_attention_scheme.json"
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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device)
    nums, tokens, labels_all = make_inputs(device)
    number_positions = NUMBER_POSITIONS.to(device)
    total = tokens.shape[0]

    condition_names = [
        "actual",
        "complete_piecewise_with_H3_soft_for_max1",
        "pure_onehot_piecewise_H3_to_ANS_for_max1",
        "pure_onehot_piecewise_H3_to_max_for_max1",
    ]
    correct = {name: 0 for name in condition_names}
    per_max_correct = {name: torch.zeros(10, dtype=torch.long) for name in condition_names}
    per_max_total = torch.zeros(10, dtype=torch.long)
    max_abs_logit_diff = {name: 0.0 for name in condition_names if name != "actual"}

    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            x = tokens[start:end]
            n = nums[start:end]
            y = labels_all[start:end]
            batch = end - start
            batch_idx = torch.arange(batch, device=device)
            is_max_slot = n == y[:, None]

            logits, _ = model(x)
            actual_digit_logits = logits[:, 10, :10]
            actual_pred = actual_digit_logits.argmax(dim=1)
            correct["actual"] += int((actual_pred == y).sum())

            seq_len = x.shape[1]
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            resid = model.tok_embed(x) + model.pos_embed(positions)
            ans_resid = resid[:, 10, :]
            layer = model.layers[0]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)

            self_values = []
            max_values = []
            actual_values = []

            for head in layer.heads:
                out, attn = head(resid, mask)
                attn_row = attn[:, 10, :]
                source_values = resid @ head.W_V.weight.detach().T
                number_attn = attn_row[:, number_positions]
                max_attn = number_attn.masked_fill(~is_max_slot, -1.0)
                max_slot = max_attn.argmax(dim=1)
                max_pos = number_positions[max_slot]

                self_values.append(source_values[:, 10, :])
                max_values.append(source_values[batch_idx, max_pos])
                actual_values.append(out[:, 10, :])

            h0_choice = torch.where((y == 9).unsqueeze(1), max_values[0], self_values[0])
            h1_choice = self_values[1]
            h2_choice = torch.where((y >= 7).unsqueeze(1), max_values[2], self_values[2])
            h3_soft_for_max1 = torch.where(
                (y == 1).unsqueeze(1),
                actual_values[3],
                torch.where((y == 0).unsqueeze(1), self_values[3], max_values[3]),
            )
            h3_ans_for_max1 = torch.where(
                (y <= 1).unsqueeze(1),
                self_values[3],
                max_values[3],
            )
            h3_max_for_max1 = torch.where(
                (y == 0).unsqueeze(1),
                self_values[3],
                max_values[3],
            )

            conditions = {
                "complete_piecewise_with_H3_soft_for_max1": [
                    h0_choice,
                    h1_choice,
                    h2_choice,
                    h3_soft_for_max1,
                ],
                "pure_onehot_piecewise_H3_to_ANS_for_max1": [
                    h0_choice,
                    h1_choice,
                    h2_choice,
                    h3_ans_for_max1,
                ],
                "pure_onehot_piecewise_H3_to_max_for_max1": [
                    h0_choice,
                    h1_choice,
                    h2_choice,
                    h3_max_for_max1,
                ],
            }

            for name, head_values in conditions.items():
                intervention_logits = digit_logits(
                    ans_resid + layer.W_O(torch.cat(head_values, dim=-1)),
                    model,
                )
                pred = intervention_logits.argmax(dim=1)
                correct[name] += int((pred == y).sum())
                max_abs_logit_diff[name] = max(
                    max_abs_logit_diff[name],
                    float((actual_digit_logits - intervention_logits).abs().max()),
                )

            for true_max in range(10):
                row_mask = y == true_max
                per_max_total[true_max] += int(row_mask.sum())
                per_max_correct["actual"][true_max] += int(
                    (actual_pred[row_mask] == y[row_mask]).sum()
                )
                for name, head_values in conditions.items():
                    intervention_logits = digit_logits(
                        ans_resid[row_mask]
                        + layer.W_O(torch.cat([value[row_mask] for value in head_values], dim=-1)),
                        model,
                    )
                    pred = intervention_logits.argmax(dim=1)
                    per_max_correct[name][true_max] += int((pred == y[row_mask]).sum())

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
            for true_max in range(10)
        }
        for name in condition_names
    }

    table = {
        "0": {"H0": "ANS", "H1": "ANS", "H2": "ANS", "H3": "ANS"},
        "1": {"H0": "ANS", "H1": "ANS", "H2": "ANS", "H3": "actual soft: ANS + max-1 positions"},
        "2": {"H0": "ANS", "H1": "ANS", "H2": "ANS", "H3": "max token"},
        "3": {"H0": "ANS", "H1": "ANS", "H2": "ANS", "H3": "max token"},
        "4": {"H0": "ANS", "H1": "ANS", "H2": "ANS", "H3": "max token"},
        "5": {"H0": "ANS", "H1": "ANS", "H2": "ANS", "H3": "max token"},
        "6": {"H0": "ANS", "H1": "ANS", "H2": "ANS", "H3": "max token"},
        "7": {"H0": "ANS", "H1": "ANS", "H2": "max token", "H3": "max token"},
        "8": {"H0": "ANS", "H1": "ANS", "H2": "max token", "H3": "max token"},
        "9": {"H0": "max token", "H1": "ANS", "H2": "max token", "H3": "max token"},
    }

    summary = {
        "scope": "all 10^5 possible inputs",
        "n_inputs": total,
        "condition_accuracy": accuracy,
        "condition_accuracy_by_true_max": accuracy_by_true_max,
        "max_abs_digit_logit_diff_vs_actual": max_abs_logit_diff,
        "counts_by_true_max": {str(i): int(per_max_total[i]) for i in range(10)},
        "attention_abstraction_table": table,
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
        for true_max in range(10):
            row = accuracy_by_true_max[name][str(true_max)]
            print(f"{name},{true_max},{row['accuracy']:.6f},{row['correct']}/{row['total']}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
