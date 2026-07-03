#!/usr/bin/env python3
"""Evaluate whether H3's OV contribution alone predicts the max token."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_h3_ov_logit_accuracy.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_h3_ov_logit_accuracy.json"
BATCH_SIZE = 4096
NUM_POSITIONS = torch.tensor([1, 3, 5, 7, 9])


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
    nums = torch.cartesian_prod(*[torch.arange(2, 10) for _ in range(5)]).to(device)
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


def tokenize(nums: list[int], device: str) -> torch.Tensor:
    return torch.tensor(
        [[10, nums[0], 11, nums[1], 11, nums[2], 11, nums[3], 11, nums[4], 12]],
        dtype=torch.long,
        device=device,
    )


def digit_logits(vec: torch.Tensor, model) -> torch.Tensor:
    return vec @ model.unembed.weight.detach()[:10].T


def vocab_logits(vec: torch.Tensor, model) -> torch.Tensor:
    return vec @ model.unembed.weight.detach().T


@torch.no_grad()
def h3_decomposition(model, tokens: torch.Tensor, nums: torch.Tensor):
    device = tokens.device
    batch, seq_len = tokens.shape
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    resid = model.tok_embed(tokens) + model.pos_embed(positions)
    ans_resid = resid[:, 10, :]
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)

    layer = model.layers[0]
    head_values = []
    head_attns = []
    for head in layer.heads:
        out, attn = head(resid, mask)
        head_values.append(out[:, 10, :])
        head_attns.append(attn[:, 10, :])

    h3 = layer.heads[3]
    h3_values_by_source = resid @ h3.W_V.weight.detach().T
    h3_attn_row = head_attns[3]

    labels = nums.max(dim=1).values
    number_positions = NUM_POSITIONS.to(device)
    number_attn = h3_attn_row[:, number_positions]
    is_max_slot = nums == labels[:, None]
    max_attn = number_attn.masked_fill(~is_max_slot, -1.0)
    max_slot_idx = max_attn.argmax(dim=1)
    max_token_positions = number_positions[max_slot_idx]

    h3_actual_value = head_values[3]
    h3_onehot_max_value = h3_values_by_source[torch.arange(batch, device=device), max_token_positions]

    d_head = h3.d_head
    w_o = layer.W_O.weight.detach()
    h3_w_o = w_o[:, 3 * d_head : 4 * d_head]
    h3_actual_out = h3_actual_value @ h3_w_o.T
    h3_onehot_out = h3_onehot_max_value @ h3_w_o.T

    concat_actual = torch.cat(head_values, dim=-1)
    concat_replace_h3 = torch.cat(head_values[:3] + [h3_onehot_max_value], dim=-1)
    final_actual = ans_resid + layer.W_O(concat_actual)
    final_replace_h3 = ans_resid + layer.W_O(concat_replace_h3)

    return {
        "labels": labels,
        "max_token_positions": max_token_positions,
        "h3_top_is_max": is_max_slot[
            torch.arange(batch, device=device),
            ((h3_attn_row.argmax(dim=1) - 1) // 2).clamp(min=0, max=4),
        ]
        & (h3_attn_row.argmax(dim=1).unsqueeze(1) == number_positions).any(dim=1),
        "h3_max_attention_mass": (number_attn * is_max_slot.float()).sum(dim=1),
        "h3_actual_digit_logits": digit_logits(h3_actual_out, model),
        "h3_onehot_digit_logits": digit_logits(h3_onehot_out, model),
        "h3_actual_vocab_logits": vocab_logits(h3_actual_out, model),
        "h3_onehot_vocab_logits": vocab_logits(h3_onehot_out, model),
        "resid_h3_actual_digit_logits": digit_logits(ans_resid + h3_actual_out, model),
        "resid_h3_onehot_digit_logits": digit_logits(ans_resid + h3_onehot_out, model),
        "final_actual_digit_logits": digit_logits(final_actual, model),
        "final_replace_h3_digit_logits": digit_logits(final_replace_h3, model),
    }


def update_counts(counts, name: str, logits: torch.Tensor, labels: torch.Tensor) -> None:
    pred = logits.argmax(dim=1)
    counts[name]["correct"] += int((pred == labels).sum())
    counts[name]["total"] += int(labels.numel())


def add_confusion(confusion: torch.Tensor, preds: torch.Tensor, labels: torch.Tensor) -> None:
    for true, pred in zip(labels.cpu(), preds.cpu(), strict=True):
        confusion[int(true), int(pred)] += 1


def summarize_examples(model, device: str):
    examples = [
        [6, 8, 4, 7, 5],
        [2, 3, 4, 5, 6],
        [9, 2, 3, 4, 5],
        [7, 7, 2, 3, 4],
        [2, 2, 2, 2, 2],
        [8, 9, 7, 6, 5],
    ]
    rows = []
    for nums_list in examples:
        nums = torch.tensor([nums_list], dtype=torch.long, device=device)
        tokens = tokenize(nums_list, device)
        decomp = h3_decomposition(model, tokens, nums)
        rows.append(
            {
                "nums": nums_list,
                "max": int(decomp["labels"][0]),
                "max_token_position_used": int(decomp["max_token_positions"][0]),
                "h3_max_attention_mass": float(decomp["h3_max_attention_mass"][0]),
                "h3_actual_pred": int(decomp["h3_actual_digit_logits"].argmax(dim=1)[0]),
                "h3_onehot_pred": int(decomp["h3_onehot_digit_logits"].argmax(dim=1)[0]),
                "resid_h3_onehot_pred": int(decomp["resid_h3_onehot_digit_logits"].argmax(dim=1)[0]),
                "final_pred": int(decomp["final_actual_digit_logits"].argmax(dim=1)[0]),
            }
        )
    return rows


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device)
    nums, tokens = make_inputs(device)
    total = tokens.shape[0]

    count_names = [
        "H3 actual OV only",
        "H3 one-hot max OV only",
        "resid + H3 actual",
        "resid + H3 one-hot max",
        "full model actual",
        "full model with H3 one-hot max",
    ]
    counts = {name: {"correct": 0, "total": 0} for name in count_names}
    vocab_counts = {
        "H3 actual OV only": {"correct": 0, "total": 0},
        "H3 one-hot max OV only": {"correct": 0, "total": 0},
    }
    h3_top_is_max_count = 0
    h3_max_attention_masses = []
    confusion_h3_onehot = torch.zeros(10, 10, dtype=torch.long)
    per_true_correct = {name: torch.zeros(10, dtype=torch.long) for name in count_names}
    per_true_total = torch.zeros(10, dtype=torch.long)

    for start in range(0, total, BATCH_SIZE):
        x = tokens[start : start + BATCH_SIZE]
        n = nums[start : start + BATCH_SIZE]
        decomp = h3_decomposition(model, x, n)
        labels = decomp["labels"]

        digit_conditions = {
            "H3 actual OV only": decomp["h3_actual_digit_logits"],
            "H3 one-hot max OV only": decomp["h3_onehot_digit_logits"],
            "resid + H3 actual": decomp["resid_h3_actual_digit_logits"],
            "resid + H3 one-hot max": decomp["resid_h3_onehot_digit_logits"],
            "full model actual": decomp["final_actual_digit_logits"],
            "full model with H3 one-hot max": decomp["final_replace_h3_digit_logits"],
        }
        for name, logits in digit_conditions.items():
            update_counts(counts, name, logits, labels)
            pred = logits.argmax(dim=1)
            for true_value in range(2, 10):
                mask = labels == true_value
                per_true_total[true_value] += int(mask.sum()) if name == count_names[0] else 0
                per_true_correct[name][true_value] += int((pred[mask] == labels[mask]).sum())

        for name, logits_key in [
            ("H3 actual OV only", "h3_actual_vocab_logits"),
            ("H3 one-hot max OV only", "h3_onehot_vocab_logits"),
        ]:
            update_counts(vocab_counts, name, decomp[logits_key], labels)

        add_confusion(
            confusion_h3_onehot,
            decomp["h3_onehot_digit_logits"].argmax(dim=1),
            labels,
        )
        h3_top_is_max_count += int(decomp["h3_top_is_max"].sum())
        h3_max_attention_masses.append(decomp["h3_max_attention_mass"].detach().cpu())

    h3_max_attention_masses_t = torch.cat(h3_max_attention_masses)
    sample_examples = summarize_examples(model, device)

    summary = {
        "n_inputs_all_numbers_ge_2": total,
        "digit_argmax_accuracy": {
            name: {
                "accuracy": counts[name]["correct"] / counts[name]["total"],
                "correct": counts[name]["correct"],
                "total": counts[name]["total"],
            }
            for name in count_names
        },
        "vocab_argmax_accuracy": {
            name: {
                "accuracy": vocab_counts[name]["correct"] / vocab_counts[name]["total"],
                "correct": vocab_counts[name]["correct"],
                "total": vocab_counts[name]["total"],
            }
            for name in vocab_counts
        },
        "h3_attention": {
            "top_key_is_a_max_token_count": h3_top_is_max_count,
            "top_key_is_a_max_token_fraction": h3_top_is_max_count / total,
            "max_token_attention_mass_mean": float(h3_max_attention_masses_t.mean()),
            "max_token_attention_mass_min": float(h3_max_attention_masses_t.min()),
            "max_token_attention_mass_max": float(h3_max_attention_masses_t.max()),
        },
        "per_true_max_digit_accuracy": {
            name: {
                str(true_value): (
                    int(per_true_correct[name][true_value]) / int(per_true_total[true_value])
                )
                for true_value in range(2, 10)
            }
            for name in count_names
        },
        "h3_onehot_digit_confusion_rows_true_cols_pred": confusion_h3_onehot.tolist(),
        "sample_examples": sample_examples,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    accuracies = [summary["digit_argmax_accuracy"][name]["accuracy"] for name in count_names]
    colors = ["#f97316", "#fb923c", "#14b8a6", "#2dd4bf", "#2563eb", "#60a5fa"]
    ax.bar(range(len(count_names)), accuracies, color=colors)
    ax.set_xticks(range(len(count_names)))
    ax.set_xticklabels(count_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Digit argmax accuracy")
    ax.set_title("Does H3's OV contribution alone predict the answer?")
    ax.grid(axis="y", alpha=0.25)
    for idx, acc in enumerate(accuracies):
        ax.text(idx, acc + 0.02, f"{acc:.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[0, 1]
    im = ax.imshow(confusion_h3_onehot[2:10].numpy(), cmap="Blues", aspect="auto")
    ax.set_yticks(range(8))
    ax.set_yticklabels(range(2, 10))
    ax.set_xticks(range(10))
    ax.set_xlabel("Predicted digit from H3 one-hot OV logits")
    ax.set_ylabel("True max")
    ax.set_title("Pure H3 one-hot OV confusion")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    true_values = list(range(2, 10))
    for name, style in [
        ("H3 one-hot max OV only", ("#f97316", "o")),
        ("resid + H3 one-hot max", ("#14b8a6", "s")),
        ("full model with H3 one-hot max", ("#2563eb", "^")),
    ]:
        color, marker = style
        values = [summary["per_true_max_digit_accuracy"][name][str(v)] for v in true_values]
        ax.plot(true_values, values, marker=marker, color=color, label=name)
    ax.set_xticks(true_values)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("True max")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by true max value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.hist(h3_max_attention_masses_t.numpy(), bins=50, color="#7c3aed", alpha=0.85)
    ax.set_xlabel("Total H3 attention mass on max-valued token positions")
    ax.set_ylabel("Input count")
    ax.set_title(
        "H3 usually reads the max token(s): "
        f"mean={summary['h3_attention']['max_token_attention_mass_mean']:.6f}"
    )
    ax.grid(alpha=0.25)

    fig.suptitle("Model 1: H3 OV logits on all inputs with every number >= 2", fontsize=15)
    fig.savefig(OUT, dpi=180)

    print(f"n_inputs_all_numbers_ge_2,{total}")
    print("condition,digit_accuracy,correct,total")
    for name in count_names:
        row = summary["digit_argmax_accuracy"][name]
        print(f"{name},{row['accuracy']:.6f},{row['correct']},{row['total']}")
    print("condition,vocab_accuracy,correct,total")
    for name in vocab_counts:
        row = summary["vocab_argmax_accuracy"][name]
        print(f"{name},{row['accuracy']:.6f},{row['correct']},{row['total']}")
    print(
        "h3_attention,"
        f"top_key_is_max={h3_top_is_max_count}/{total},"
        f"mean_max_mass={summary['h3_attention']['max_token_attention_mass_mean']:.6f},"
        f"min_max_mass={summary['h3_attention']['max_token_attention_mass_min']:.6f}"
    )
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
