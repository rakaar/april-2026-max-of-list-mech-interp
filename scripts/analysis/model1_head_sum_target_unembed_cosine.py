#!/usr/bin/env python3
"""Measure summed head-output cosine with the true-max unembedding."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_head_sum_target_unembed_cosine.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_head_sum_target_unembed_cosine.json"
COMPARISON_OUT = ROOT / "docs" / "assets" / "model1_head_sum_target_vs_nonmax_unembed_cosine.png"
COMPARISON_JSON_OUT = ROOT / "docs" / "assets" / "model1_head_sum_target_vs_nonmax_unembed_cosine.json"
ACCURACY_JSON_OUT = ROOT / "docs" / "assets" / "model1_head_sum_cosine_logit_accuracy.json"
ACCURACY_OUT = ROOT / "docs" / "assets" / "model1_head_sum_norm_weighted_vs_cosine_accuracy.png"
BATCH_SIZE = 4096
EXPECTED_COUNTS_BY_MAX = {
    0: 1,
    1: 31,
    2: 211,
    3: 781,
    4: 2101,
    5: 4651,
    6: 9031,
    7: 15961,
    8: 26281,
    9: 40951,
}


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


def make_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nums = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)]).to(device)
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


def plot_bar_chart(mean_cosine: torch.Tensor, std_cosine: torch.Tensor) -> None:
    max_values = list(range(10))
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    ax.bar(
        max_values,
        mean_cosine.numpy(),
        yerr=std_cosine.numpy(),
        capsize=4,
        color="#2563eb",
        alpha=0.86,
        ecolor="#111827",
        linewidth=0.8,
    )
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(max_values)
    ax.set_xlabel("True max")
    ax.set_ylabel("cosine(head sum, W_U[true max])")
    ax.set_title("Model 1: summed head output direction vs true-max unembedding")
    ax.set_ylim(-1.0, 1.05)
    ax.grid(axis="y", alpha=0.25)

    for max_value, mean, std in zip(max_values, mean_cosine.tolist(), std_cosine.tolist()):
        offset = 0.04 if mean >= 0 else -0.08
        ax.text(
            max_value,
            float(mean) + offset,
            f"{float(mean):.2f}",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=8,
        )
        if float(std) > 0.01:
            ax.text(max_value, 0.92, f"std {float(std):.2f}", ha="center", fontsize=7)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)


def plot_comparison_chart(
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    nonmax_mean: torch.Tensor,
    nonmax_std: torch.Tensor,
    nonmax_counts: torch.Tensor,
) -> None:
    max_values = torch.arange(10)
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 6.0), constrained_layout=True)

    ax.bar(
        max_values.numpy() - width / 2,
        target_mean.numpy(),
        width=width,
        yerr=target_std.numpy(),
        capsize=3,
        label="true max",
        color="#2563eb",
        alpha=0.86,
        ecolor="#111827",
        linewidth=0.8,
    )

    has_nonmax = nonmax_counts > 0
    ax.bar(
        max_values[has_nonmax].numpy() + width / 2,
        nonmax_mean[has_nonmax].numpy(),
        width=width,
        yerr=nonmax_std[has_nonmax].numpy(),
        capsize=3,
        label="query non-max avg",
        color="#dc2626",
        alpha=0.82,
        ecolor="#111827",
        linewidth=0.8,
    )

    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(max_values.numpy())
    ax.set_xlabel("True max")
    ax.set_ylabel("cosine(head sum, digit unembedding)")
    ax.set_title("Model 1: head-sum direction toward true max vs query non-max digits")
    ax.set_ylim(-1.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    for max_value in max_values.tolist():
        ax.text(
            max_value - width / 2,
            float(target_mean[max_value]) + 0.04,
            f"{float(target_mean[max_value]):.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        if bool(has_nonmax[max_value]):
            ax.text(
                max_value + width / 2,
                float(nonmax_mean[max_value]) + 0.04,
                f"{float(nonmax_mean[max_value]):.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        else:
            ax.text(
                max_value + width / 2,
                0.04,
                "n/a",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#7f1d1d",
            )

    COMPARISON_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARISON_OUT, dpi=180)
    plt.close(fig)


def plot_accuracy_chart(
    norm_weighted_accuracy: torch.Tensor,
    cosine_accuracy: torch.Tensor,
) -> None:
    max_values = torch.arange(10)
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    if not torch.allclose(norm_weighted_accuracy, norm_weighted_accuracy[0].expand_as(norm_weighted_accuracy)):
        raise ValueError(
            "norm-weighted accuracy is not constant across true-max groups; "
            f"got {norm_weighted_accuracy.tolist()}"
        )

    ax.axhline(
        float(norm_weighted_accuracy[0]),
        color="#2563eb",
        linewidth=2.4,
        alpha=0.88,
        label=r"$||W_U[d]|| \cdot \cos(h, W_U[d])$",
    )
    ax.scatter(
        max_values.numpy(),
        cosine_accuracy.numpy(),
        s=90,
        color="#dc2626",
        edgecolor="#111827",
        linewidth=0.9,
        zorder=3,
        label=r"$\cos(h, W_U[d])$ only",
    )

    ax.axhline(1.0, color="#111827", linewidth=1.0, alpha=0.45)
    ax.set_xticks(max_values.numpy())
    ax.set_xlabel("True max")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model 1: norm-weighted cosine vs cosine-only head-sum readout")
    ax.set_ylim(-0.04, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")

    for max_value, acc in zip(max_values.tolist(), cosine_accuracy.tolist()):
        if float(acc) < 0.995:
            ax.text(
                max_value,
                float(acc) + 0.045,
                f"{float(acc):.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ACCURACY_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ACCURACY_OUT, dpi=180)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    nums, tokens = make_inputs(device)
    labels = nums.max(dim=1).values

    counts = torch.bincount(labels, minlength=10).cpu()
    expected_counts = torch.tensor([EXPECTED_COUNTS_BY_MAX[i] for i in range(10)])
    if not torch.equal(counts, expected_counts):
        raise ValueError(f"unexpected true-max counts: {counts.tolist()}")

    cosine_sum = torch.zeros(10, dtype=torch.float64)
    cosine_sq_sum = torch.zeros(10, dtype=torch.float64)
    min_cosine = torch.full((10,), float("inf"), dtype=torch.float64)
    max_cosine = torch.full((10,), float("-inf"), dtype=torch.float64)
    nonmax_counts = torch.zeros(10, dtype=torch.long)
    nonmax_cosine_sum = torch.zeros(10, dtype=torch.float64)
    nonmax_cosine_sq_sum = torch.zeros(10, dtype=torch.float64)
    nonmax_min_cosine = torch.full((10,), float("inf"), dtype=torch.float64)
    nonmax_max_cosine = torch.full((10,), float("-inf"), dtype=torch.float64)
    raw_correct = torch.zeros(10, dtype=torch.long)
    norm_weighted_correct = torch.zeros(10, dtype=torch.long)
    cosine_correct = torch.zeros(10, dtype=torch.long)
    raw_pred_counts = torch.zeros((10, 10), dtype=torch.long)
    norm_weighted_pred_counts = torch.zeros((10, 10), dtype=torch.long)
    cosine_pred_counts = torch.zeros((10, 10), dtype=torch.long)
    raw_norm_weighted_mismatches = 0

    layer = model.layers[0]
    w_o = layer.W_O.weight.detach()
    u_numbers = model.unembed.weight.detach()[:10]
    u_number_norms = u_numbers.norm(dim=1)
    seq_len = tokens.shape[1]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)

    with torch.no_grad():
        for start in range(0, tokens.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, tokens.shape[0])
            batch_nums = nums[start:end]
            batch_tokens = tokens[start:end]
            batch_labels = labels[start:end]
            batch_labels_cpu = batch_labels.cpu()
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            resid = model.tok_embed(batch_tokens) + model.pos_embed(positions)
            target_u = model.unembed.weight.detach()[batch_labels]

            head_vectors = []
            for head_idx, head in enumerate(layer.heads):
                head_values, _ = head(resid, causal_mask)
                d_head = head.d_head
                w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
                head_vectors.append(head_values[:, 10, :] @ w_o_head.T)

            head_sum = sum(head_vectors)
            cosine = F.cosine_similarity(head_sum, target_u, dim=-1).cpu().to(torch.float64)
            raw_digit_logits = head_sum @ u_numbers.T
            all_digit_cosine = F.cosine_similarity(
                head_sum[:, None, :],
                u_numbers[None, :, :],
                dim=-1,
            )
            norm_weighted_scores = all_digit_cosine * u_number_norms[None, :]
            raw_pred_cpu = raw_digit_logits.argmax(dim=1).cpu()
            norm_weighted_pred_cpu = norm_weighted_scores.argmax(dim=1).cpu()
            cosine_pred_cpu = all_digit_cosine.argmax(dim=1).cpu()
            raw_norm_weighted_mismatches += int(
                (raw_pred_cpu != norm_weighted_pred_cpu).sum()
            )
            query_digit_cosine = all_digit_cosine.gather(dim=1, index=batch_nums)
            nonmax_position_mask = batch_nums != batch_labels[:, None]
            has_nonmax = nonmax_position_mask.any(dim=1)
            nonmax_position_counts = nonmax_position_mask.sum(dim=1).clamp_min(1)
            avg_nonmax_cosine = (
                (query_digit_cosine * nonmax_position_mask.float()).sum(dim=1)
                / nonmax_position_counts
            ).cpu().to(torch.float64)
            has_nonmax_cpu = has_nonmax.cpu()

            for max_value in range(10):
                mask = batch_labels_cpu == max_value
                if not bool(mask.any()):
                    continue
                selected = cosine[mask]
                cosine_sum[max_value] += selected.sum()
                cosine_sq_sum[max_value] += selected.square().sum()
                min_cosine[max_value] = torch.minimum(min_cosine[max_value], selected.min())
                max_cosine[max_value] = torch.maximum(max_cosine[max_value], selected.max())

                raw_selected = raw_pred_cpu[mask]
                norm_weighted_selected = norm_weighted_pred_cpu[mask]
                cosine_selected = cosine_pred_cpu[mask]
                raw_correct[max_value] += int((raw_selected == max_value).sum())
                norm_weighted_correct[max_value] += int(
                    (norm_weighted_selected == max_value).sum()
                )
                cosine_correct[max_value] += int((cosine_selected == max_value).sum())
                raw_pred_counts[max_value] += torch.bincount(raw_selected, minlength=10)
                norm_weighted_pred_counts[max_value] += torch.bincount(
                    norm_weighted_selected,
                    minlength=10,
                )
                cosine_pred_counts[max_value] += torch.bincount(cosine_selected, minlength=10)

                nonmax_mask = mask & has_nonmax_cpu
                if not bool(nonmax_mask.any()):
                    continue
                nonmax_selected = avg_nonmax_cosine[nonmax_mask]
                nonmax_counts[max_value] += int(nonmax_mask.sum())
                nonmax_cosine_sum[max_value] += nonmax_selected.sum()
                nonmax_cosine_sq_sum[max_value] += nonmax_selected.square().sum()
                nonmax_min_cosine[max_value] = torch.minimum(
                    nonmax_min_cosine[max_value],
                    nonmax_selected.min(),
                )
                nonmax_max_cosine[max_value] = torch.maximum(
                    nonmax_max_cosine[max_value],
                    nonmax_selected.max(),
                )

    counts_f = counts.to(torch.float64)
    mean_cosine = cosine_sum / counts_f
    numerator = cosine_sq_sum - counts_f * mean_cosine.square()
    denominator = (counts_f - 1).clamp_min(1.0)
    variance = (numerator / denominator).clamp_min(0.0)
    variance[counts <= 1] = 0.0
    std_cosine = variance.sqrt()

    nonmax_counts_f = nonmax_counts.to(torch.float64).clamp_min(1.0)
    nonmax_mean_cosine = nonmax_cosine_sum / nonmax_counts_f
    nonmax_numerator = nonmax_cosine_sq_sum - nonmax_counts_f * nonmax_mean_cosine.square()
    nonmax_denominator = (nonmax_counts_f - 1).clamp_min(1.0)
    nonmax_variance = (nonmax_numerator / nonmax_denominator).clamp_min(0.0)
    nonmax_variance[nonmax_counts <= 1] = 0.0
    nonmax_std_cosine = nonmax_variance.sqrt()
    nonmax_mean_cosine[nonmax_counts == 0] = float("nan")
    nonmax_min_cosine[nonmax_counts == 0] = float("nan")
    nonmax_max_cosine[nonmax_counts == 0] = float("nan")

    data = {
        "description": (
            "All 10^5 Model 1 inputs. head_sum is H0_vec + H1_vec + H2_vec + H3_vec, "
            "where each Hh_vec is the actual ANS-position head output after that "
            "head's W_O slice. The reported metric is cosine_similarity(head_sum, "
            "W_U[true_max])."
        ),
        "n_inputs_total": int(tokens.shape[0]),
        "sequence_format": "[BOS] n0 [SEP] n1 [SEP] n2 [SEP] n3 [SEP] n4 [ANS]",
        "counts_by_true_max": {str(i): int(counts[i]) for i in range(10)},
        "rows": [
            {
                "true_max": max_value,
                "count": int(counts[max_value]),
                "mean_cosine": float(mean_cosine[max_value]),
                "std_cosine": float(std_cosine[max_value]),
                "min_cosine": float(min_cosine[max_value]),
                "max_cosine": float(max_cosine[max_value]),
            }
            for max_value in range(10)
        ],
    }

    JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")
    plot_bar_chart(mean_cosine, std_cosine)

    comparison_data = {
        "description": (
            "All 10^5 Model 1 inputs. For each input, target_cosine compares "
            "head_sum with W_U[true_max]. query_nonmax_avg_cosine first computes "
            "cosine(head_sum, W_U[n_i]) for each query position whose value is not "
            "the true max, then averages those position-level cosines within the "
            "input. Duplicate non-max query entries count as repeated positions. "
            "Inputs with no non-max positions are excluded from the non-max summary."
        ),
        "n_inputs_total": int(tokens.shape[0]),
        "sequence_format": "[BOS] n0 [SEP] n1 [SEP] n2 [SEP] n3 [SEP] n4 [ANS]",
        "counts_by_true_max": {str(i): int(counts[i]) for i in range(10)},
        "nonmax_input_counts_by_true_max": {
            str(i): int(nonmax_counts[i]) for i in range(10)
        },
        "rows": [
            {
                "true_max": max_value,
                "target": {
                    "count": int(counts[max_value]),
                    "mean_cosine": float(mean_cosine[max_value]),
                    "std_cosine": float(std_cosine[max_value]),
                    "min_cosine": float(min_cosine[max_value]),
                    "max_cosine": float(max_cosine[max_value]),
                },
                "query_nonmax_average": (
                    None
                    if int(nonmax_counts[max_value]) == 0
                    else {
                        "count": int(nonmax_counts[max_value]),
                        "mean_cosine": float(nonmax_mean_cosine[max_value]),
                        "std_cosine": float(nonmax_std_cosine[max_value]),
                        "min_cosine": float(nonmax_min_cosine[max_value]),
                        "max_cosine": float(nonmax_max_cosine[max_value]),
                    }
                ),
            }
            for max_value in range(10)
        ],
    }
    COMPARISON_JSON_OUT.write_text(json.dumps(comparison_data, indent=2) + "\n")
    plot_comparison_chart(
        mean_cosine,
        std_cosine,
        nonmax_mean_cosine,
        nonmax_std_cosine,
        nonmax_counts,
    )

    accuracy_data = {
        "description": (
            "All 10^5 Model 1 inputs. raw_head_sum logits are head_sum @ W_U_numbers.T. "
            "norm_weighted_cosine predicts with ||W_U[d]|| * cosine(head_sum, W_U[d]), "
            "which removes only the common per-input ||head_sum|| factor. cosine_logits "
            "predicts with cosine(head_sum, W_U[d]), removing both the common head_sum "
            "norm and each digit unembedding row norm from the prediction score."
        ),
        "n_inputs_total": int(tokens.shape[0]),
        "sequence_format": "[BOS] n0 [SEP] n1 [SEP] n2 [SEP] n3 [SEP] n4 [ANS]",
        "counts_by_true_max": {str(i): int(counts[i]) for i in range(10)},
        "raw_norm_weighted_prediction_mismatches": raw_norm_weighted_mismatches,
        "rows": [
            {
                "true_max": max_value,
                "count": int(counts[max_value]),
                "raw_head_sum": {
                    "correct": int(raw_correct[max_value]),
                    "accuracy": float(raw_correct[max_value] / counts[max_value]),
                    "prediction_distribution": {
                        str(digit): int(raw_pred_counts[max_value, digit])
                        for digit in range(10)
                        if int(raw_pred_counts[max_value, digit]) > 0
                    },
                },
                "norm_weighted_cosine": {
                    "correct": int(norm_weighted_correct[max_value]),
                    "accuracy": float(
                        norm_weighted_correct[max_value] / counts[max_value]
                    ),
                    "prediction_distribution": {
                        str(digit): int(norm_weighted_pred_counts[max_value, digit])
                        for digit in range(10)
                        if int(norm_weighted_pred_counts[max_value, digit]) > 0
                    },
                },
                "cosine_logits": {
                    "correct": int(cosine_correct[max_value]),
                    "accuracy": float(cosine_correct[max_value] / counts[max_value]),
                    "prediction_distribution": {
                        str(digit): int(cosine_pred_counts[max_value, digit])
                        for digit in range(10)
                        if int(cosine_pred_counts[max_value, digit]) > 0
                    },
                },
            }
            for max_value in range(10)
        ],
    }
    ACCURACY_JSON_OUT.write_text(json.dumps(accuracy_data, indent=2) + "\n")
    norm_weighted_accuracy = norm_weighted_correct.to(torch.float64) / counts.to(torch.float64)
    cosine_accuracy = cosine_correct.to(torch.float64) / counts.to(torch.float64)
    plot_accuracy_chart(norm_weighted_accuracy, cosine_accuracy)

    print("true_max,count,mean_cosine,std_cosine,min_cosine,max_cosine")
    for max_value in range(10):
        print(
            f"{max_value},{int(counts[max_value])},"
            f"{float(mean_cosine[max_value]):.6f},"
            f"{float(std_cosine[max_value]):.6f},"
            f"{float(min_cosine[max_value]):.6f},"
            f"{float(max_cosine[max_value]):.6f}"
        )
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")
    print("target_vs_query_nonmax")
    print("true_max,target_count,target_mean,target_std,nonmax_count,nonmax_mean,nonmax_std")
    for max_value in range(10):
        nonmax_mean = (
            "nan"
            if int(nonmax_counts[max_value]) == 0
            else f"{float(nonmax_mean_cosine[max_value]):.6f}"
        )
        nonmax_std = (
            "nan"
            if int(nonmax_counts[max_value]) == 0
            else f"{float(nonmax_std_cosine[max_value]):.6f}"
        )
        print(
            f"{max_value},{int(counts[max_value])},"
            f"{float(mean_cosine[max_value]):.6f},"
            f"{float(std_cosine[max_value]):.6f},"
            f"{int(nonmax_counts[max_value])},"
            f"{nonmax_mean},{nonmax_std}"
        )
    print(f"wrote,{COMPARISON_OUT}")
    print(f"wrote,{COMPARISON_JSON_OUT}")
    print("cosine_logit_accuracy")
    print(
        "true_max,count,norm_weighted_cosine_acc,cosine_acc,"
        "cosine_correct,cosine_pred_distribution"
    )
    for max_value in range(10):
        cosine_distribution = {
            digit: int(cosine_pred_counts[max_value, digit])
            for digit in range(10)
            if int(cosine_pred_counts[max_value, digit]) > 0
        }
        print(
            f"{max_value},{int(counts[max_value])},"
            f"{float(norm_weighted_correct[max_value] / counts[max_value]):.6f},"
            f"{float(cosine_correct[max_value] / counts[max_value]):.6f},"
            f"{int(cosine_correct[max_value])},"
            f"{cosine_distribution}"
        )
    print(f"raw_norm_weighted_prediction_mismatches,{raw_norm_weighted_mismatches}")
    print(f"wrote,{ACCURACY_OUT}")
    print(f"wrote,{ACCURACY_JSON_OUT}")


if __name__ == "__main__":
    main()
