#!/usr/bin/env python3
"""Compare residual-only, head-sum-only, and full logits on all inputs."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_residual_vs_head_sum_accuracy.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_residual_vs_head_sum_accuracy.json"


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
    variants = ["residual_only", "head_sum_only", "full"]

    counts: dict[int, int] = defaultdict(int)
    correct = {variant: defaultdict(int) for variant in variants}
    pred_counts = {(max_value, variant): Counter() for max_value in range(10) for variant in variants}
    logit_sums = {
        (max_value, variant): torch.zeros(10)
        for max_value in range(10)
        for variant in variants
    }

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

            head_sum = sum(head_vectors)
            logits = {
                "residual_only": number_logits(ans_resid, model),
                "head_sum_only": number_logits(head_sum, model),
                "full": number_logits(ans_resid + head_sum, model),
            }

            for max_value in range(10):
                mask_value = labels == max_value
                count = int(mask_value.sum())
                if count == 0:
                    continue
                counts[max_value] += count
                for variant in variants:
                    variant_logits = logits[variant]
                    pred = variant_logits.argmax(dim=1)
                    correct[variant][max_value] += int((pred[mask_value] == labels[mask_value]).sum())
                    pred_counts[(max_value, variant)].update(pred[mask_value].tolist())
                    logit_sums[(max_value, variant)] += variant_logits[mask_value].sum(dim=0).cpu()

    rows = []
    for max_value in range(10):
        row = {"true_max": max_value, "count": counts[max_value], "variants": {}}
        for variant in variants:
            row["variants"][variant] = {
                "accuracy": correct[variant][max_value] / counts[max_value],
                "correct": correct[variant][max_value],
                "prediction_distribution": {
                    str(k): v for k, v in sorted(pred_counts[(max_value, variant)].items())
                },
                "avg_logits": [
                    float(x) for x in (logit_sums[(max_value, variant)] / counts[max_value]).tolist()
                ],
            }
        rows.append(row)

    result = {
        "description": (
            "All 10^5 inputs. residual_only uses ans_resid @ W_U_numbers.T; "
            "head_sum_only uses (H0+H1+H2+H3) @ W_U_numbers.T; full uses their sum."
        ),
        "variants": variants,
        "rows": rows,
    }
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    xs = list(range(10))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    width = 0.25
    offsets = {"residual_only": -width, "head_sum_only": 0.0, "full": width}
    colors = {"residual_only": "#c75b4a", "head_sum_only": "#377eb8", "full": "#4daf4a"}
    labels = {
        "residual_only": "residual only",
        "head_sum_only": "head sum only",
        "full": "residual + heads",
    }
    for variant in variants:
        accs = [rows[max_value]["variants"][variant]["accuracy"] for max_value in xs]
        axes[0].bar(
            [x + offsets[variant] for x in xs],
            accs,
            width=width,
            label=labels[variant],
            color=colors[variant],
        )
    axes[0].set_xticks(xs)
    axes[0].set_xlabel("true max")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Accuracy by true max")
    axes[0].legend(loc="lower right")

    head_preds = []
    residual_preds = []
    for max_value in xs:
        residual_dist = rows[max_value]["variants"]["residual_only"]["prediction_distribution"]
        head_dist = rows[max_value]["variants"]["head_sum_only"]["prediction_distribution"]
        residual_preds.append(next(iter(residual_dist.keys())))
        head_preds.append(next(iter(head_dist.keys())))
    axes[1].plot(xs, xs, color="#222222", linewidth=1, label="correct")
    axes[1].scatter(xs, [int(x) for x in residual_preds], color=colors["residual_only"], s=70, label="residual only")
    axes[1].scatter(xs, [int(x) for x in head_preds], color=colors["head_sum_only"], s=70, marker="x", label="head sum only")
    for x, residual_pred, head_pred in zip(xs, residual_preds, head_preds):
        axes[1].text(x, int(residual_pred) + 0.15, residual_pred, ha="center", fontsize=8, color=colors["residual_only"])
        axes[1].text(x, int(head_pred) - 0.35, head_pred, ha="center", fontsize=8, color=colors["head_sum_only"])
    axes[1].set_xticks(xs)
    axes[1].set_yticks(xs)
    axes[1].set_xlabel("true max")
    axes[1].set_ylabel("predicted digit")
    axes[1].set_title("Predicted digit by true max")
    axes[1].legend(loc="upper left")

    fig.suptitle("Model 1: heads alone solve max-of-list; ANS residual alone predicts 7")
    fig.savefig(OUT, dpi=180)

    print("max,count,residual_only_acc,head_sum_only_acc,full_acc,residual_pred,head_sum_pred")
    for row in rows:
        max_value = row["true_max"]
        residual_dist = row["variants"]["residual_only"]["prediction_distribution"]
        head_dist = row["variants"]["head_sum_only"]["prediction_distribution"]
        print(
            f"{max_value},{row['count']},"
            f"{row['variants']['residual_only']['accuracy']:.6f},"
            f"{row['variants']['head_sum_only']['accuracy']:.6f},"
            f"{row['variants']['full']['accuracy']:.6f},"
            f"{residual_dist},{head_dist}"
        )
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
