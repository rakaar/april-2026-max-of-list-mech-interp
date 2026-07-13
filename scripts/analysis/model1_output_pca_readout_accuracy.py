#!/usr/bin/env python3
"""Test max-task readout in PCs fitted to the centered total output matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
JSON_OUT = ROOT / "docs" / "assets" / "model1_output_pca_readout_accuracy.json"
PNG_OUT = ROOT / "docs" / "assets" / "model1_output_pca_readout_accuracy.png"
HF_REPO = "andyrdt/04_2026_puzzle_1a"
BATCH_SIZE = 4096
KS = (1, 2, 3)


def load_model():
    model_py_path = hf_hub_download(HF_REPO, "model.py")
    spec = importlib.util.spec_from_file_location("model", model_py_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config_path = hf_hub_download(HF_REPO, "config.json")
    weights_path = hf_hub_download(HF_REPO, "model.pt")
    config = json.loads(Path(config_path).read_text())["model"]

    model = module.AttentionOnlyTransformer.from_config(config)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, config


def tokenize(numbers: torch.Tensor) -> torch.Tensor:
    tokens = torch.empty((numbers.shape[0], 11), dtype=torch.long, device=numbers.device)
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
    return tokens


def extract_ans_head_values(model, tokens: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    residual = model.tok_embed(tokens) + model.pos_embed(positions)
    causal_mask = torch.tril(
        torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device)
    ).unsqueeze(0)
    values = []
    for head in model.layers[0].heads:
        head_values, _ = head(residual, causal_mask)
        values.append(head_values[:, 10, :])
    return torch.stack(values, dim=1)


def fit_output_pca(output_matrix: torch.Tensor) -> dict:
    output_mean = output_matrix.mean(dim=0, keepdim=True)
    centered = output_matrix - output_mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    explained = energy / energy.sum()
    covariance = centered.T @ centered / (centered.shape[0] - 1)
    covariance_variance = torch.linalg.eigvalsh(covariance).flip(0)
    svd_variance = energy / (centered.shape[0] - 1)
    if not torch.allclose(
        covariance_variance, svd_variance, rtol=1e-9, atol=1e-11
    ):
        raise AssertionError("covariance eigenvalues and squared singular values disagree")
    identity = torch.eye(vh.shape[0], dtype=vh.dtype)
    if not torch.allclose(vh @ vh.T, identity, rtol=1e-9, atol=1e-10):
        raise AssertionError("output PCA directions are not orthonormal")
    return {
        "mean": output_mean.squeeze(0),
        "centered": centered,
        "singular_values": singular_values,
        "directions": vh,
        "explained": explained,
        "rank": int(torch.linalg.matrix_rank(centered)),
        "total_centered_variance": float(torch.trace(covariance)),
    }


def variance_captured_by_output_pcs(
    matrix: torch.Tensor, directions: torch.Tensor
) -> dict:
    matrix = matrix.detach().cpu().double()
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    total_energy = centered.square().sum()
    cumulative_fraction = {}
    for k in KS:
        basis = directions[:k].T
        projected = centered @ basis
        cumulative_fraction[str(k)] = float(projected.square().sum() / total_energy)
    return {
        "matrix_shape": list(matrix.shape),
        "centering": "subtract the mean row of this unembedding matrix",
        "cumulative_fraction_captured": cumulative_fraction,
    }


def empty_readout_stats(vocab_size: int) -> dict:
    return {
        "correct": 0,
        "prediction_counts": torch.zeros(vocab_size, dtype=torch.long),
        "correct_by_true_max": torch.zeros(10, dtype=torch.long),
        "special_token_prediction_count": 0,
    }


def update_readout_stats(stats: dict, prediction: torch.Tensor, labels: torch.Tensor) -> None:
    prediction_cpu = prediction.detach().cpu()
    labels_cpu = labels.detach().cpu()
    stats["correct"] += int((prediction_cpu == labels_cpu).sum())
    stats["prediction_counts"] += torch.bincount(
        prediction_cpu, minlength=len(stats["prediction_counts"])
    )
    for true_max in range(10):
        mask = labels_cpu == true_max
        stats["correct_by_true_max"][true_max] += int(
            (prediction_cpu[mask] == labels_cpu[mask]).sum()
        )
    stats["special_token_prediction_count"] += int((prediction_cpu >= 10).sum())


def serializable_readout(
    stats: dict, total: int, counts_by_true_max: torch.Tensor
) -> dict:
    return {
        "accuracy": stats["correct"] / total,
        "correct": stats["correct"],
        "total": total,
        "prediction_distribution": {
            str(token): int(count)
            for token, count in enumerate(stats["prediction_counts"])
            if int(count) > 0
        },
        "special_token_prediction_count": stats["special_token_prediction_count"],
        "accuracy_by_true_max": {
            str(true_max): {
                "accuracy": (
                    int(stats["correct_by_true_max"][true_max])
                    / int(counts_by_true_max[true_max])
                ),
                "correct": int(stats["correct_by_true_max"][true_max]),
                "total": int(counts_by_true_max[true_max]),
            }
            for true_max in range(10)
        },
    }


def evaluate(model, output_matrices: torch.Tensor, pca: dict) -> dict:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    all_numbers = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)])
    all_labels = all_numbers.max(dim=1).values
    counts_by_true_max = torch.bincount(all_labels, minlength=10)
    total = len(all_numbers)

    vocabularies = {"full_vocabulary": model.unembed.weight.detach()}
    bases = {
        k: pca["directions"][:k].T.to(device=device, dtype=dtype) for k in KS
    }
    output_matrices_device = output_matrices.to(device=device, dtype=dtype)
    reduced_outputs = {
        k: output_matrices_device @ basis for k, basis in bases.items()
    }
    low_unembeddings = {}
    raw_low_unembeddings = {}
    for vocab_name, unembedding in vocabularies.items():
        unembedding = unembedding.to(device=device, dtype=dtype)
        centered = unembedding - unembedding.mean(dim=0, keepdim=True)
        low_unembeddings[vocab_name] = {
            k: centered @ basis for k, basis in bases.items()
        }
        raw_low_unembeddings[vocab_name] = {
            k: unembedding @ basis for k, basis in bases.items()
        }

    baseline_stats = {
        name: empty_readout_stats(len(unembedding))
        for name, unembedding in vocabularies.items()
    }
    low_stats = {
        name: {k: empty_readout_stats(len(unembedding)) for k in KS}
        for name, unembedding in vocabularies.items()
    }
    max_coordinate_error = {k: 0.0 for k in KS}
    centered_raw_argmax_mismatches = {
        name: {k: 0 for k in KS} for name in vocabularies
    }

    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            numbers = all_numbers[start:end].to(device)
            labels = all_labels[start:end].to(device)
            head_values = extract_ans_head_values(model, tokenize(numbers))
            full_head_writes = torch.einsum(
                "bhd,hdr->bhr", head_values, output_matrices_device
            )
            full_head_sum = full_head_writes.sum(dim=1)

            for vocab_name, unembedding in vocabularies.items():
                logits = full_head_sum @ unembedding.to(device=device, dtype=dtype).T
                update_readout_stats(baseline_stats[vocab_name], logits.argmax(dim=1), labels)

            for k in KS:
                low_head_writes = torch.einsum(
                    "bhd,hdk->bhk", head_values, reduced_outputs[k]
                )
                low_head_sum = low_head_writes.sum(dim=1)
                reference = full_head_sum @ bases[k]
                error = float((low_head_sum - reference).abs().max())
                max_coordinate_error[k] = max(max_coordinate_error[k], error)
                if not torch.allclose(
                    low_head_sum, reference, rtol=1e-5, atol=2e-4
                ):
                    raise AssertionError(
                        f"per-head and sum-then-project coordinates disagree for k={k}: {error}"
                    )

                for vocab_name in vocabularies:
                    centered_logits = (
                        low_head_sum @ low_unembeddings[vocab_name][k].T
                    )
                    raw_logits = (
                        low_head_sum @ raw_low_unembeddings[vocab_name][k].T
                    )
                    centered_prediction = centered_logits.argmax(dim=1)
                    raw_prediction = raw_logits.argmax(dim=1)
                    centered_raw_argmax_mismatches[vocab_name][k] += int(
                        (centered_prediction != raw_prediction).sum()
                    )
                    update_readout_stats(
                        low_stats[vocab_name][k], centered_prediction, labels
                    )

    baseline = {
        name: serializable_readout(stats, total, counts_by_true_max)
        for name, stats in baseline_stats.items()
    }
    if any(row["correct"] != total for row in baseline.values()):
        raise AssertionError(f"full 64D head-sum baseline is not perfect: {baseline}")
    if baseline["full_vocabulary"]["special_token_prediction_count"] != 0:
        raise AssertionError("full 64D baseline predicted a special token")

    readout = {
        name: {
            str(k): serializable_readout(low_stats[name][k], total, counts_by_true_max)
            for k in KS
        }
        for name in vocabularies
    }
    if any(
        centered_raw_argmax_mismatches[name][k] != 0
        for name in vocabularies
        for k in KS
    ):
        raise AssertionError(
            "centering the projected unembedding changed an argmax: "
            f"{centered_raw_argmax_mismatches}"
        )

    return {
        "n_inputs": total,
        "counts_by_true_max": {
            str(true_max): int(counts_by_true_max[true_max])
            for true_max in range(10)
        },
        "full_64d_head_sum_baseline": baseline,
        "output_pca_readout": readout,
        "max_abs_coordinate_error_per_head_vs_sum_then_project": {
            str(k): max_coordinate_error[k] for k in KS
        },
        "centered_vs_raw_projected_unembedding_argmax_mismatches": {
            name: {str(k): count for k, count in rows.items()}
            for name, rows in centered_raw_argmax_mismatches.items()
        },
    }


def plot_results(result: dict) -> None:
    explained = np.asarray(result["output_matrix_pca"]["explained_variance"])
    cumulative = np.cumsum(explained)
    xs = np.asarray(KS)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    axes[0].bar(xs, 100.0 * explained[:3], color=["#2563eb", "#16a34a", "#dc2626"])
    axes[0].plot(xs, 100.0 * cumulative[:3], color="#111827", marker="o", label="cumulative")
    for k in KS:
        axes[0].text(
            k,
            100.0 * cumulative[k - 1] + 1.5,
            f"{100.0 * cumulative[k - 1]:.2f}%",
            ha="center",
            weight="bold",
        )
    axes[0].set_xticks(xs)
    axes[0].set_ylim(0.0, 108.0)
    axes[0].set_xlabel("Number of centered output-matrix PCs")
    axes[0].set_ylabel("Centered row variance (%)")
    axes[0].set_title("Output-matrix PCA variance")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)

    full_accuracies = [
        result["evaluation"]["output_pca_readout"]["full_vocabulary"][str(k)]["accuracy"]
        for k in KS
    ]
    axes[1].plot(
        xs,
        100.0 * np.asarray(full_accuracies),
        color="#7c3aed",
        marker="o",
        linewidth=2.0,
        label="14-way full vocabulary",
    )
    for k, accuracy in zip(KS, full_accuracies):
        axes[1].text(
            k,
            100.0 * accuracy + 1.4,
            f"{100.0 * accuracy:.2f}%",
            color="#7c3aed",
            ha="center",
            fontsize=9,
        )
    axes[1].axhline(100.0, color="#64748b", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(xs)
    axes[1].set_xlim(0.85, 3.20)
    axes[1].set_ylim(0.0, 108.0)
    axes[1].set_xlabel("Number of centered output-matrix PCs")
    axes[1].set_ylabel("Accuracy over 100,000 inputs (%)")
    axes[1].set_title("Direct readout in the output-PCA basis")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle("Model 1: reverse-basis test using PCs of the total output matrix")
    fig.savefig(PNG_OUT, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    torch.manual_seed(0)
    model, config = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    layer = model.layers[0]
    stored_w_o = layer.W_O.weight.detach().cpu().double()
    output_matrices = torch.stack(
        [
            stored_w_o[:, head_idx * head.d_head : (head_idx + 1) * head.d_head].T
            for head_idx, head in enumerate(layer.heads)
        ]
    )
    output_matrix = output_matrices.flatten(0, 1)
    if output_matrices.shape != (4, 16, 64) or output_matrix.shape != (64, 64):
        raise AssertionError(
            f"unexpected output shapes: {output_matrices.shape}, {output_matrix.shape}"
        )
    if not torch.equal(output_matrix, stored_w_o.T):
        raise AssertionError("stacked mathematical output map does not equal W_O.weight.T")

    pca = fit_output_pca(output_matrix)
    evaluation = evaluate(model, output_matrices, pca)
    explained = pca["explained"]
    unembedding_variance = {
        "description": (
            "Centered unembedding row variance captured by the same output-matrix "
            "PCA basis used for the reduced readout."
        ),
        "formula": (
            "fraction_k = ||U_centered @ Q_k||_F^2 / ||U_centered||_F^2, "
            "where Q_k contains the top k output-matrix PCs"
        ),
        "full_vocabulary": variance_captured_by_output_pcs(
            model.unembed.weight.detach(), pca["directions"]
        ),
    }

    result = {
        "description": (
            "Centered PCA is fitted to the 64 rows of the mathematical total output map "
            "O_all = stack(O_h) = stored W_O.weight.T. Actual post-attention [ANS] head "
            "values are mapped through O_h @ Q_k for k=1,2,3, summed in k dimensions, "
            "and scored against unembeddings projected into the same output-PCA basis."
        ),
        "hf_repo": HF_REPO,
        "model_config": config,
        "device": str(device),
        "matrix_orientation": {
            "stored_W_O_weight_shape": list(stored_w_o.shape),
            "per_head_mathematical_output_shape": [16, 64],
            "total_mathematical_output_shape": [64, 64],
            "identity": "O_all = stack(O_0,O_1,O_2,O_3) = stored W_O.weight.T",
        },
        "output_matrix_pca": {
            "centering": "subtract the mean of the 64 output-direction rows only when fitting PCA",
            "rank_after_centering": pca["rank"],
            "total_centered_variance": pca["total_centered_variance"],
            "singular_values": [float(value) for value in pca["singular_values"]],
            "explained_variance": [float(value) for value in explained],
            "cumulative_explained_variance": [
                float(value) for value in torch.cumsum(explained, dim=0)
            ],
            "top3_basis_rows": [
                [float(value) for value in row] for row in pca["directions"][:3]
            ],
        },
        "readout_formula": {
            "basis": "Q_k = top k columns from centered O_all PCA, shape 64 x k",
            "reduced_output": "O_h_low = O_h @ Q_k, shape 16 x k",
            "head_write": "z_h = value_h @ O_h_low, shape batch x k",
            "sum": "z = sum_h z_h, shape batch x k",
            "unembedding": "U_low = (W_U - mean_rows(W_U)) @ Q_k, shape vocab x k",
            "logits": "relative_logits = z @ U_low.T",
            "output_mean_policy": (
                "The O_all row mean is not subtracted from actual head writes and is not "
                "added back; accuracy tests the pure linear Q_k subspace."
            ),
        },
        "unembedding_variance_in_output_pcs": unembedding_variance,
        "evaluation": evaluation,
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot_results(result)

    print(
        "k,output_cumulative_variance,full_vocab_unembedding_variance,"
        "full_vocab_accuracy,special_predictions"
    )
    for k in KS:
        full_row = evaluation["output_pca_readout"]["full_vocabulary"][str(k)]
        print(
            f"{k},{float(torch.cumsum(explained, dim=0)[k - 1]):.12f},"
            f"{unembedding_variance['full_vocabulary']['cumulative_fraction_captured'][str(k)]:.12f},"
            f"{full_row['accuracy']:.12f},"
            f"{full_row['special_token_prediction_count']}"
        )
    print(f"wrote,{JSON_OUT}")
    print(f"wrote,{PNG_OUT}")


if __name__ == "__main__":
    main()
