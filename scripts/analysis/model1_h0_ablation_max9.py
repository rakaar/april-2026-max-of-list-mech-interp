#!/usr/bin/env python3
"""Ablate H0 to test whether it is necessary for true-max-9 predictions."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
JSON_OUT = ROOT / "docs" / "assets" / "model1_h0_ablation_max9.json"


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
    all_nums = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)])

    summary = defaultdict(lambda: defaultdict(int))
    pred_distributions = defaultdict(Counter)
    margin_sums = defaultdict(lambda: defaultdict(float))
    margin_counts = defaultdict(int)
    chunk_size = 4096

    with torch.no_grad():
        for start in range(0, len(all_nums), chunk_size):
            nums_t = all_nums[start : start + chunk_size]
            labels = nums_t.max(dim=1).values
            tokens = torch.tensor([tokenize(row.tolist()) for row in nums_t], dtype=torch.long)
            batch, seq_len = tokens.shape

            positions = torch.arange(seq_len).unsqueeze(0)
            resid = model.tok_embed(tokens) + model.pos_embed(positions)
            ans_resid = resid[:, 10, :]

            layer = model.layers[0]
            w_o = layer.W_O.weight.detach()
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

            head_vectors = []
            for head_idx, head in enumerate(layer.heads):
                head_values, _ = head(resid, mask)
                d_head = head.d_head
                w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
                head_vectors.append(head_values[:, 10, :] @ w_o_head.T)

            final_vec = ans_resid + sum(head_vectors)
            zero_h0_vec = ans_resid + head_vectors[1] + head_vectors[2] + head_vectors[3]

            final_logits = number_logits(final_vec, model)
            h0_logits = number_logits(head_vectors[0], model)
            zero_h0_logits = number_logits(zero_h0_vec, model)

            # Less mechanistically clean, but it tests the literal target-logit question:
            # remove only H0's contribution to output digit 9.
            zero_h0_logit9_only = final_logits.clone()
            zero_h0_logit9_only[:, 9] -= h0_logits[:, 9]

            predictions = {
                "baseline": final_logits.argmax(dim=1),
                "zero_H0_vector": zero_h0_logits.argmax(dim=1),
                "zero_H0_logit9_only": zero_h0_logit9_only.argmax(dim=1),
            }

            contains8 = (nums_t == 8).any(dim=1)
            groups = {
                "all": torch.ones(batch, dtype=torch.bool),
                "max8": labels == 8,
                "max9": labels == 9,
                "max9_contains8": (labels == 9) & contains8,
                "max9_no8": (labels == 9) & (~contains8),
            }

            margins = {
                "baseline": final_logits[:, 9] - final_logits[:, 8],
                "H0_component": h0_logits[:, 9] - h0_logits[:, 8],
                "zero_H0_vector": zero_h0_logits[:, 9] - zero_h0_logits[:, 8],
            }

            for group_name, group_mask in groups.items():
                count = int(group_mask.sum())
                if count == 0:
                    continue
                summary[group_name]["count"] += count

                for intervention_name, pred in predictions.items():
                    correct = int((pred[group_mask] == labels[group_mask]).sum())
                    summary[group_name][f"{intervention_name}_correct"] += correct
                    pred_distributions[(group_name, intervention_name)].update(
                        pred[group_mask].tolist()
                    )

                for margin_name, margin in margins.items():
                    margin_sums[group_name][margin_name] += float(margin[group_mask].sum())
                margin_counts[group_name] += count

    groups_out = {}
    for group_name in ["all", "max8", "max9", "max9_contains8", "max9_no8"]:
        count = summary[group_name]["count"]
        groups_out[group_name] = {
            "count": count,
            "accuracy": {},
            "correct": {},
            "prediction_distribution": {},
            "avg_logit9_minus_logit8_margin": {},
        }
        for intervention_name in ["baseline", "zero_H0_vector", "zero_H0_logit9_only"]:
            correct = summary[group_name][f"{intervention_name}_correct"]
            groups_out[group_name]["correct"][intervention_name] = correct
            groups_out[group_name]["accuracy"][intervention_name] = correct / count
            groups_out[group_name]["prediction_distribution"][intervention_name] = {
                str(k): v
                for k, v in sorted(
                    pred_distributions[(group_name, intervention_name)].items()
                )
            }
        for margin_name in ["baseline", "H0_component", "zero_H0_vector"]:
            groups_out[group_name]["avg_logit9_minus_logit8_margin"][margin_name] = (
                margin_sums[group_name][margin_name] / margin_counts[group_name]
            )

    result = {
        "description": (
            "Full-space H0 ablation test. zero_H0_vector removes the H0 output vector "
            "from the final ANS residual before unembedding. zero_H0_logit9_only removes "
            "only H0's direct contribution to output digit 9."
        ),
        "groups": groups_out,
    }
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    print("accuracy")
    print("group,count,baseline,zero_H0_vector,zero_H0_logit9_only")
    for group_name, group in groups_out.items():
        print(
            group_name,
            group["count"],
            f"{group['accuracy']['baseline']:.6f}",
            f"{group['accuracy']['zero_H0_vector']:.6f}",
            f"{group['accuracy']['zero_H0_logit9_only']:.6f}",
            sep=",",
        )

    print("\nmax9 prediction distributions")
    for group_name in ["max9", "max9_contains8", "max9_no8"]:
        group = groups_out[group_name]
        print(group_name, "zero_H0_vector", group["prediction_distribution"]["zero_H0_vector"], sep=",")
        print(
            group_name,
            "zero_H0_logit9_only",
            group["prediction_distribution"]["zero_H0_logit9_only"],
            sep=",",
        )

    print("\navg logit9-logit8 margins")
    print("group,baseline,H0_component,zero_H0_vector")
    for group_name in ["max8", "max9", "max9_contains8", "max9_no8"]:
        margins = groups_out[group_name]["avg_logit9_minus_logit8_margin"]
        print(
            group_name,
            f"{margins['baseline']:+.6f}",
            f"{margins['H0_component']:+.6f}",
            f"{margins['zero_H0_vector']:+.6f}",
            sep=",",
        )
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
