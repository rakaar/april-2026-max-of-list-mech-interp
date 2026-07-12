#!/usr/bin/env python3
"""Test cosine-only versus dot-product readout in the top-3 digit PCA space."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_lowdim_cosine_readout.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_lowdim_cosine_readout.json"
BATCH_SIZE = 4096
EXPECTED_COUNTS = torch.tensor([1, 31, 211, 781, 2101, 4651, 9031, 15961, 26281, 40951])


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
    return model, config


def make_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numbers = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)]).to(device)
    labels = numbers.max(dim=1).values
    tokens = torch.empty((numbers.shape[0], 11), dtype=torch.long, device=device)
    tokens[:, 0] = 10
    tokens[:, 1] = numbers[:, 0]
    tokens[:, 2] = 11
    tokens[:, 3] = numbers[:, 1]
    tokens[:, 4] = 11
    tokens[:, 5] = numbers[:, 2]
    tokens[:, 6] = 11
    tokens[:, 7] = numbers[:, 3]
    tokens[:, 8] = 11
    tokens[:, 9] = numbers[:, 4]
    tokens[:, 10] = 12
    return numbers, tokens, labels


def sample_std(sum_values: torch.Tensor, sum_squares: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    counts_f = counts.to(torch.float64)
    means = sum_values / counts_f
    numerator = sum_squares - counts_f * means.square()
    denominator = (counts_f - 1).clamp_min(1.0)
    variance = (numerator / denominator).clamp_min(0.0)
    variance[counts <= 1] = 0.0
    return variance.sqrt()


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model()
    model = model.to(device)
    _, tokens, labels = make_inputs(device)
    counts = torch.bincount(labels, minlength=10).cpu()
    if not torch.equal(counts, EXPECTED_COUNTS):
        raise AssertionError(f"unexpected counts by true maximum: {counts.tolist()}")

    digit_unembedding = model.unembed.weight.detach()[:10]
    digit_mean = digit_unembedding.mean(dim=0)
    centered_digits = digit_unembedding - digit_mean
    _, singular_values, directions = torch.linalg.svd(centered_digits, full_matrices=False)
    basis = directions[:3]
    digit_coordinates = centered_digits @ basis.T
    digit_norms = digit_coordinates.norm(dim=1)
    explained = singular_values.square() / singular_values.square().sum()

    dot_correct = torch.zeros(10, dtype=torch.long)
    cosine_correct = torch.zeros(10, dtype=torch.long)
    cosine_confusion = torch.zeros((10, 10), dtype=torch.long)
    dot_confusion = torch.zeros((10, 10), dtype=torch.long)
    target_cosine_sum = torch.zeros(10, dtype=torch.float64)
    target_cosine_sq_sum = torch.zeros(10, dtype=torch.float64)
    target_cosine_min = torch.full((10,), float("inf"), dtype=torch.float64)
    target_cosine_max = torch.full((10,), float("-inf"), dtype=torch.float64)
    cosine_margin_sum = torch.zeros(10, dtype=torch.float64)
    dot_norm_weighted_mismatches = 0
    max0_details = None

    layer = model.layers[0]
    w_o = layer.W_O.weight.detach()
    seq_len = tokens.shape[1]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)

    with torch.no_grad():
        for start in range(0, tokens.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, tokens.shape[0])
            batch_tokens = tokens[start:end]
            batch_labels = labels[start:end]
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            residual = model.tok_embed(batch_tokens) + model.pos_embed(positions)

            head_vectors = []
            for head_idx, head in enumerate(layer.heads):
                head_values, _ = head(residual, causal_mask)
                d_head = head.d_head
                output_matrix = w_o[
                    :, head_idx * d_head : (head_idx + 1) * d_head
                ].T
                head_vectors.append(head_values[:, 10, :] @ output_matrix)

            head_sum = sum(head_vectors)
            head_coordinates = head_sum @ basis.T
            dot_scores = head_coordinates @ digit_coordinates.T
            cosine_scores = F.cosine_similarity(
                head_coordinates[:, None, :],
                digit_coordinates[None, :, :],
                dim=-1,
            )
            norm_weighted_scores = cosine_scores * digit_norms[None, :]

            dot_pred = dot_scores.argmax(dim=1)
            cosine_pred = cosine_scores.argmax(dim=1)
            norm_weighted_pred = norm_weighted_scores.argmax(dim=1)
            dot_norm_weighted_mismatches += int((dot_pred != norm_weighted_pred).sum())
            cosine_top2 = torch.topk(cosine_scores, 2, dim=1)
            cosine_margin = cosine_top2.values[:, 0] - cosine_top2.values[:, 1]
            target_cosine = cosine_scores.gather(1, batch_labels[:, None]).squeeze(1)

            batch_labels_cpu = batch_labels.cpu()
            dot_pred_cpu = dot_pred.cpu()
            cosine_pred_cpu = cosine_pred.cpu()
            target_cosine_cpu = target_cosine.cpu().to(torch.float64)
            cosine_margin_cpu = cosine_margin.cpu().to(torch.float64)

            for true_max in range(10):
                mask = batch_labels_cpu == true_max
                if not bool(mask.any()):
                    continue
                dot_selected = dot_pred_cpu[mask]
                cosine_selected = cosine_pred_cpu[mask]
                target_selected = target_cosine_cpu[mask]
                dot_correct[true_max] += int((dot_selected == true_max).sum())
                cosine_correct[true_max] += int((cosine_selected == true_max).sum())
                dot_confusion[true_max] += torch.bincount(dot_selected, minlength=10)
                cosine_confusion[true_max] += torch.bincount(cosine_selected, minlength=10)
                target_cosine_sum[true_max] += target_selected.sum()
                target_cosine_sq_sum[true_max] += target_selected.square().sum()
                target_cosine_min[true_max] = torch.minimum(
                    target_cosine_min[true_max], target_selected.min()
                )
                target_cosine_max[true_max] = torch.maximum(
                    target_cosine_max[true_max], target_selected.max()
                )
                cosine_margin_sum[true_max] += cosine_margin_cpu[mask].sum()

            if start == 0:
                max0_index = int((batch_labels == 0).nonzero(as_tuple=False)[0])
                max0_cosines = cosine_scores[max0_index]
                max0_dots = dot_scores[max0_index]
                max0_details = {
                    "head_sum_coordinates": [
                        float(value) for value in head_coordinates[max0_index]
                    ],
                    "cosine_by_digit": [float(value) for value in max0_cosines],
                    "dot_product_by_digit": [float(value) for value in max0_dots],
                    "cosine_prediction": int(max0_cosines.argmax()),
                    "dot_product_prediction": int(max0_dots.argmax()),
                }

    if max0_details is None:
        raise AssertionError("max-0 details were not collected")
    if int(dot_correct.sum()) != 100000:
        raise AssertionError(f"3d dot readout was not perfect: {int(dot_correct.sum())}/100000")
    if dot_norm_weighted_mismatches != 0:
        raise AssertionError(
            "norm-weighted cosine and dot-product predictions differ on "
            f"{dot_norm_weighted_mismatches} inputs"
        )

    counts_f = counts.to(torch.float64)
    target_cosine_mean = target_cosine_sum / counts_f
    target_cosine_std = sample_std(target_cosine_sum, target_cosine_sq_sum, counts)
    cosine_accuracy = cosine_correct.to(torch.float64) / counts_f
    dot_accuracy = dot_correct.to(torch.float64) / counts_f
    mean_cosine_winner_margin = cosine_margin_sum / counts_f
    normalized_confusion = cosine_confusion.to(torch.float64) / counts_f[:, None]

    overall_cosine_correct = int(cosine_correct.sum())
    overall_cosine_accuracy = overall_cosine_correct / 100000
    macro_cosine_accuracy = float(cosine_accuracy.mean())
    perfect_cosine_groups = [
        true_max for true_max in range(10) if int(cosine_correct[true_max]) == int(counts[true_max])
    ]

    rows = []
    for true_max in range(10):
        rows.append(
            {
                "true_max": true_max,
                "count": int(counts[true_max]),
                "dot_product": {
                    "correct": int(dot_correct[true_max]),
                    "accuracy": float(dot_accuracy[true_max]),
                    "prediction_distribution": {
                        str(digit): int(dot_confusion[true_max, digit])
                        for digit in range(10)
                        if int(dot_confusion[true_max, digit]) > 0
                    },
                },
                "cosine_only": {
                    "correct": int(cosine_correct[true_max]),
                    "accuracy": float(cosine_accuracy[true_max]),
                    "prediction_distribution": {
                        str(digit): int(cosine_confusion[true_max, digit])
                        for digit in range(10)
                        if int(cosine_confusion[true_max, digit]) > 0
                    },
                    "mean_target_cosine": float(target_cosine_mean[true_max]),
                    "std_target_cosine": float(target_cosine_std[true_max]),
                    "min_target_cosine": float(target_cosine_min[true_max]),
                    "max_target_cosine": float(target_cosine_max[true_max]),
                    "mean_winner_margin": float(mean_cosine_winner_margin[true_max]),
                },
            }
        )

    data = {
        "description": (
            "All 100000 Model 1 inputs. The actual four-head ANS output sum is projected "
            "onto the top three PCA directions of the centered digit unembeddings. "
            "dot_product predicts argmax_d z dot U3[d]. cosine_only normalizes both z "
            "and each U3[d]. norm_weighted_cosine multiplies cosine by ||U3[d]|| and "
            "must match dot_product because ||z|| is common across candidate digits."
        ),
        "hf_repo": "andyrdt/04_2026_puzzle_1a",
        "model_config": config,
        "n_inputs_total": 100000,
        "sequence_format": "[BOS] n0 [SEP] n1 [SEP] n2 [SEP] n3 [SEP] n4 [ANS]",
        "counts_by_true_max": {str(i): int(counts[i]) for i in range(10)},
        "digit_pca_explained_variance": [float(value) for value in explained[:3]],
        "digit_pca_top3_cumulative_variance": float(explained[:3].sum()),
        "digit_coordinates": [
            [float(value) for value in row] for row in digit_coordinates
        ],
        "digit_coordinate_norms": [float(value) for value in digit_norms],
        "overall": {
            "dot_product_accuracy": 1.0,
            "norm_weighted_cosine_accuracy": 1.0,
            "cosine_only_correct": overall_cosine_correct,
            "cosine_only_accuracy": overall_cosine_accuracy,
            "macro_cosine_only_accuracy": macro_cosine_accuracy,
            "perfect_cosine_only_true_max_groups": perfect_cosine_groups,
            "dot_vs_norm_weighted_prediction_mismatches": dot_norm_weighted_mismatches,
        },
        "rows": rows,
        "cosine_confusion_counts": cosine_confusion.tolist(),
        "cosine_confusion_row_normalized": normalized_confusion.tolist(),
        "max0_details": max0_details,
    }
    JSON_OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8), constrained_layout=True)
    digits = np.arange(10)
    cosine_accuracy_np = cosine_accuracy.numpy()
    bar_colors = ["#0f766e" if value == 1.0 else "#dc2626" for value in cosine_accuracy_np]
    axes[0].bar(digits, cosine_accuracy_np, color=bar_colors, alpha=0.9)
    axes[0].axhline(1.0, color="#111827", linewidth=1.2)
    axes[0].set_xticks(digits)
    axes[0].set_ylim(0.0, 1.1)
    axes[0].set_xlabel("True maximum")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(
        f"Cosine-only accuracy by true max\noverall {overall_cosine_accuracy:.3%}; "
        "dot product 100%"
    )
    axes[0].grid(axis="y", alpha=0.22)
    for digit, value in enumerate(cosine_accuracy_np):
        axes[0].text(
            digit,
            min(value + 0.035, 1.035),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    im = axes[1].imshow(
        normalized_confusion.numpy(),
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    axes[1].set_xticks(digits)
    axes[1].set_yticks(digits)
    axes[1].set_xlabel("Cosine-only prediction")
    axes[1].set_ylabel("True maximum")
    axes[1].set_title("Row-normalized cosine confusion")
    for row in range(10):
        for col in range(10):
            value = float(normalized_confusion[row, col])
            if value >= 0.01:
                axes[1].text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "#111827",
                    fontsize=7,
                )
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Fraction")

    norm_colors = plt.get_cmap("tab10")(digits)
    axes[2].bar(digits, digit_norms.cpu().numpy(), color=norm_colors, alpha=0.9)
    axes[2].set_xticks(digits)
    axes[2].set_xlabel("Digit")
    axes[2].set_ylabel("Norm in top-3 digit PCA space")
    axes[2].set_title("Projected unembedding magnitudes")
    axes[2].grid(axis="y", alpha=0.22)
    for digit, value in enumerate(digit_norms.cpu().tolist()):
        axes[2].text(
            digit,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    fig.suptitle(
        "Model 1: cosine alone is not the three-dimensional digit readout",
        fontsize=15,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)

    print(
        "true_max,count,dot_accuracy,cosine_accuracy,cosine_correct,"
        "cosine_prediction_distribution,target_cosine_mean,target_cosine_std"
    )
    for row in rows:
        print(
            f"{row['true_max']},{row['count']},{row['dot_product']['accuracy']:.6f},"
            f"{row['cosine_only']['accuracy']:.6f},{row['cosine_only']['correct']},"
            f"{row['cosine_only']['prediction_distribution']},"
            f"{row['cosine_only']['mean_target_cosine']:.6f},"
            f"{row['cosine_only']['std_target_cosine']:.6f}"
        )
    print(f"overall_cosine_accuracy,{overall_cosine_accuracy:.6f}")
    print(f"macro_cosine_accuracy,{macro_cosine_accuracy:.6f}")
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
