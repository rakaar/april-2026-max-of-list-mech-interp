#!/usr/bin/env python3
"""Test the max-task readout in PCs fitted to the full unembedding matrix."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from model1_output_pca_readout_accuracy import (
    BATCH_SIZE,
    KS,
    extract_ans_head_values,
    load_model,
    tokenize,
)


ROOT = Path(__file__).resolve().parents[2]
JSON_OUT = ROOT / "docs" / "assets" / "model1_unembedding_pca_readout_accuracy.json"


def fit_centered_pca(matrix: torch.Tensor) -> dict:
    matrix = matrix.detach().cpu().double()
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    explained = energy / energy.sum()
    return {
        "centered": centered,
        "basis": vh.T,
        "singular_values": singular_values,
        "explained": explained,
        "rank": int(torch.linalg.matrix_rank(centered)),
    }


def per_head_output_matrices(model) -> torch.Tensor:
    layer = model.layers[0]
    stored = layer.W_O.weight.detach().cpu().double()
    matrices = torch.stack(
        [
            stored[:, head_idx * head.d_head : (head_idx + 1) * head.d_head].T
            for head_idx, head in enumerate(layer.heads)
        ]
    )
    if matrices.shape != (4, 16, 64):
        raise AssertionError(f"unexpected per-head output shape: {matrices.shape}")
    if not torch.equal(matrices.flatten(0, 1), stored.T):
        raise AssertionError("stacked per-head maps do not equal stored W_O.weight.T")
    return matrices


def captured_fractions(centered: torch.Tensor, basis: torch.Tensor) -> dict:
    total_energy = centered.square().sum()
    return {
        str(k): float((centered @ basis[:, :k]).square().sum() / total_energy)
        for k in KS
    }


def evaluate(model, matrices: torch.Tensor, basis: torch.Tensor) -> dict:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    numbers = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)])
    labels = numbers.max(dim=1).values
    total = len(numbers)

    basis_device = {
        k: basis[:, :k].to(device=device, dtype=dtype) for k in KS
    }
    matrices_device = matrices.to(device=device, dtype=dtype)
    reduced_outputs = {
        k: matrices_device @ basis_device[k] for k in KS
    }
    unembedding = model.unembed.weight.detach()
    centered_unembedding = unembedding - unembedding.mean(dim=0, keepdim=True)
    low_unembedding = {
        k: centered_unembedding @ basis_device[k] for k in KS
    }
    raw_low_unembedding = {
        k: unembedding @ basis_device[k] for k in KS
    }

    correct = {k: 0 for k in KS}
    prediction_counts = {
        k: torch.zeros(len(unembedding), dtype=torch.long) for k in KS
    }
    special_predictions = {k: 0 for k in KS}
    correct_by_max = {k: torch.zeros(10, dtype=torch.long) for k in KS}
    counts_by_max = torch.bincount(labels, minlength=10)
    baseline_correct = 0
    max_coordinate_error = {k: 0.0 for k in KS}
    centered_raw_mismatches = {k: 0 for k in KS}

    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_numbers = numbers[start:end].to(device)
            batch_labels = labels[start:end].to(device)
            head_values = extract_ans_head_values(model, tokenize(batch_numbers))
            head_writes = torch.einsum(
                "bhd,hdr->bhr", head_values, matrices_device
            )
            head_sum = head_writes.sum(dim=1)
            baseline_prediction = (head_sum @ unembedding.T).argmax(dim=1)
            baseline_correct += int((baseline_prediction == batch_labels).sum())

            for k in KS:
                low_heads = torch.einsum(
                    "bhd,hdk->bhk", head_values, reduced_outputs[k]
                )
                low_sum = low_heads.sum(dim=1)
                reference = head_sum @ basis_device[k]
                error = float((low_sum - reference).abs().max())
                max_coordinate_error[k] = max(max_coordinate_error[k], error)
                if not torch.allclose(low_sum, reference, rtol=1e-5, atol=2e-4):
                    raise AssertionError(
                        f"per-head and sum-then-project routes disagree for k={k}: {error}"
                    )

                logits = low_sum @ low_unembedding[k].T
                raw_logits = low_sum @ raw_low_unembedding[k].T
                prediction = logits.argmax(dim=1)
                raw_prediction = raw_logits.argmax(dim=1)
                centered_raw_mismatches[k] += int(
                    (prediction != raw_prediction).sum()
                )
                prediction_cpu = prediction.cpu()
                labels_cpu = batch_labels.cpu()
                correct[k] += int((prediction_cpu == labels_cpu).sum())
                prediction_counts[k] += torch.bincount(
                    prediction_cpu, minlength=len(unembedding)
                )
                special_predictions[k] += int((prediction_cpu >= 10).sum())
                for true_max in range(10):
                    mask = labels_cpu == true_max
                    correct_by_max[k][true_max] += int(
                        (prediction_cpu[mask] == labels_cpu[mask]).sum()
                    )

    if baseline_correct != total:
        raise AssertionError(
            f"full 64D attention-head sum is not perfect: {baseline_correct}/{total}"
        )
    if any(centered_raw_mismatches.values()):
        raise AssertionError(
            f"centering unembeddings changed projected argmaxes: {centered_raw_mismatches}"
        )
    if correct[3] != total or special_predictions[3] != 0:
        raise AssertionError(
            f"three-PC readout failed: {correct[3]}/{total}, special={special_predictions[3]}"
        )

    return {
        "n_inputs": total,
        "full_64d_head_sum_correct": baseline_correct,
        "counts_by_true_max": {
            str(true_max): int(counts_by_max[true_max]) for true_max in range(10)
        },
        "by_k": {
            str(k): {
                "accuracy": correct[k] / total,
                "correct": correct[k],
                "total": total,
                "special_token_prediction_count": special_predictions[k],
                "prediction_distribution": {
                    str(token): int(count)
                    for token, count in enumerate(prediction_counts[k])
                    if int(count) > 0
                },
                "accuracy_by_true_max": {
                    str(true_max): {
                        "accuracy": int(correct_by_max[k][true_max])
                        / int(counts_by_max[true_max]),
                        "correct": int(correct_by_max[k][true_max]),
                        "total": int(counts_by_max[true_max]),
                    }
                    for true_max in range(10)
                },
                "max_abs_coordinate_error_per_head_vs_sum_then_project": (
                    max_coordinate_error[k]
                ),
                "centered_vs_raw_unembedding_argmax_mismatches": (
                    centered_raw_mismatches[k]
                ),
            }
            for k in KS
        },
    }


def alignment(unembedding_basis: torch.Tensor, output_pca: dict) -> dict:
    q_u = unembedding_basis[:, :3].clone()
    q_o = output_pca["basis"][:, :3].clone()
    signs = []
    for pc_idx in range(3):
        sign = 1.0 if float(q_u[:, pc_idx] @ q_o[:, pc_idx]) >= 0.0 else -1.0
        q_o[:, pc_idx] *= sign
        signs.append(sign)
    cosine_matrix = q_u.T @ q_o
    overlap_singular_values = torch.linalg.svdvals(cosine_matrix)
    principal_angles = torch.rad2deg(
        torch.arccos(overlap_singular_values.clamp(-1.0, 1.0))
    )
    return {
        "definition": "C = Q_U.T @ Q_O; rows are full-vocabulary W_U PCs and columns are output-matrix PCs",
        "output_pc_signs_applied": signs,
        "pc_cosine_matrix": [
            [float(value) for value in row] for row in cosine_matrix
        ],
        "same_index_pc_cosines": [
            float(value) for value in torch.diagonal(cosine_matrix)
        ],
        "cosines_of_principal_angles": [
            float(value) for value in overlap_singular_values
        ],
        "principal_angles_degrees": [
            float(value) for value in principal_angles
        ],
    }


def main() -> None:
    torch.manual_seed(0)
    model, config = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    matrices = per_head_output_matrices(model)
    output_matrix = matrices.flatten(0, 1)
    unembedding = model.unembed.weight.detach().cpu().double()
    unembedding_pca = fit_centered_pca(unembedding)
    output_pca = fit_centered_pca(output_matrix)
    q_u = unembedding_pca["basis"]

    unembedding_capture = {
        str(k): float(unembedding_pca["explained"][:k].sum()) for k in KS
    }
    output_capture = captured_fractions(output_pca["centered"], q_u)
    evaluation = evaluate(model, matrices, q_u)
    pc_alignment = alignment(q_u, output_pca)

    result = {
        "description": (
            "PCA is fitted to the centered rows of the full 14x64 unembedding. "
            "The resulting Q_k basis reduces each 16x64 per-head output map to "
            "16xk and is used for a full 14-token readout over all 100000 inputs."
        ),
        "hf_repo": "andyrdt/04_2026_puzzle_1a",
        "model_config": config,
        "device": str(device),
        "basis": {
            "source": "centered full-vocabulary unembedding rows",
            "source_shape": [14, 64],
            "centered_rank": unembedding_pca["rank"],
            "basis_shape": [64, 3],
            "explained_variance_by_pc": [
                float(value) for value in unembedding_pca["explained"][:3]
            ],
            "cumulative_unembedding_variance_captured": unembedding_capture,
        },
        "output_matrix_variance_in_unembedding_pcs": {
            "matrix_shape": [64, 64],
            "centering": "subtract the mean of the 64 output-direction rows",
            "cumulative_fraction_captured": output_capture,
        },
        "readout": {
            "reduced_output": "W_O_low^h = W_O^h @ Q_k, shape 16 x k",
            "head_write": "z_h = V_h @ W_O_low^h, shape batch x k",
            "sum": "z = sum_h z_h, shape batch x k",
            "unembedding": "U_low = (U - mean_vocab(U)) @ Q_k, shape 14 x k",
            "logits": "z @ U_low.T, shape batch x 14",
        },
        "evaluation": evaluation,
        "alignment_with_output_pcs": pc_alignment,
        "related_digit_only_alignment": {
            "page": "docs/2026-07-12.md",
            "json": "docs/assets/model1_output_unembedding_pc_alignment.json",
            "note": (
                "The earlier interactive used the ten digit-unembedding rows. "
                "This artifact recomputes alignment for the full 14-token basis "
                "used by the accuracy table."
            ),
        },
    }
    JSON_OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print("k,unembedding_variance,output_variance,full_vocab_accuracy,special_predictions")
    for k in KS:
        row = evaluation["by_k"][str(k)]
        print(
            f"{k},{unembedding_capture[str(k)]:.12f},{output_capture[str(k)]:.12f},"
            f"{row['accuracy']:.12f},{row['special_token_prediction_count']}"
        )
    print("pc_cosine_matrix")
    for row in pc_alignment["pc_cosine_matrix"]:
        print(",".join(f"{value:+.9f}" for value in row))
    print(
        "principal_angles_degrees,"
        + ",".join(
            f"{value:.9f}" for value in pc_alignment["principal_angles_degrees"]
        )
    )
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
