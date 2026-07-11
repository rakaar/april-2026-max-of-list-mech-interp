#!/usr/bin/env python3
"""Project actual per-head ANS writes into the unembedding PCA bases."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_head_outputs_in_unembedding_pca_3d.png"
OVERLAY_OUT = ROOT / "docs" / "assets" / "model1_head_outputs_with_unembedding_pca_3d.png"
READOUT_OUT = ROOT / "docs" / "assets" / "model1_direct_lowdim_unembedding_readout_accuracy.png"
PER_HEAD_READOUT_OUT = ROOT / "docs" / "assets" / "model1_per_head_lowdim_output_accuracy.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_head_outputs_in_unembedding_pca.json"
BATCH_SIZE = 4096
SEED = 0
SELECTED_MAXIMA = [1, 3, 5, 7, 9]
EXPECTED_EXAMPLES = [
    [1, 0, 0, 1, 1],
    [3, 0, 1, 2, 1],
    [2, 5, 1, 3, 4],
    [7, 7, 4, 0, 1],
    [9, 2, 1, 4, 2],
]
COMPONENT_LABELS = ["H0", "H1", "H2", "H3", "sum"]
COMPONENT_COLORS = ["#2563eb", "#f59e0b", "#16a34a", "#dc2626", "#111827"]


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


def tokenize(nums: torch.Tensor) -> torch.Tensor:
    tokens = torch.empty((nums.shape[0], 11), dtype=torch.long, device=nums.device)
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
    return tokens


def select_examples() -> torch.Tensor:
    all_nums = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)])
    labels = all_nums.max(dim=1).values
    generator = torch.Generator().manual_seed(SEED)
    selected = []
    for true_max in SELECTED_MAXIMA:
        indices = (labels == true_max).nonzero(as_tuple=False).flatten()
        selected_idx = indices[torch.randperm(len(indices), generator=generator)[0]]
        selected.append(all_nums[selected_idx])
    examples = torch.stack(selected)
    if examples.tolist() != EXPECTED_EXAMPLES:
        raise AssertionError(f"seeded examples changed: {examples.tolist()}")
    if examples.max(dim=1).values.tolist() != SELECTED_MAXIMA:
        raise AssertionError("selected examples do not have the requested maxima")
    return examples


def fit_pca(token_vectors: torch.Tensor) -> dict[str, torch.Tensor | int]:
    mean = token_vectors.mean(dim=0)
    centered = token_vectors - mean
    _, singular_values, directions = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    explained = energy / energy.sum()
    rank = int(torch.linalg.matrix_rank(centered))
    identity = torch.eye(directions.shape[0], dtype=directions.dtype)
    if not torch.allclose(directions @ directions.T, identity, atol=1e-5, rtol=1e-5):
        raise AssertionError("PCA directions are not orthonormal")
    return {
        "mean": mean,
        "centered": centered,
        "directions": directions,
        "singular_values": singular_values,
        "explained": explained,
        "rank": rank,
    }


def extract_component_vectors(model, tokens: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    resid = model.tok_embed(tokens) + model.pos_embed(positions)
    layer = model.layers[0]
    w_o = layer.W_O.weight.detach()
    causal_mask = torch.tril(
        torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device)
    ).unsqueeze(0)

    head_vectors = []
    for head_idx, head in enumerate(layer.heads):
        head_values, _ = head(resid, causal_mask)
        d_head = head.d_head
        w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
        head_vectors.append(head_values[:, 10, :] @ w_o_head.T)

    heads = torch.stack(head_vectors, dim=1)
    head_sum = heads.sum(dim=1, keepdim=True)
    components = torch.cat([heads, head_sum], dim=1)
    if components.shape != (tokens.shape[0], 5, model.d_model):
        raise AssertionError(f"unexpected component shape: {components.shape}")
    if not torch.allclose(components[:, :4].sum(dim=1), components[:, 4], atol=1e-5, rtol=1e-5):
        raise AssertionError("head sum does not equal H0 + H1 + H2 + H3")
    return components


def extract_ans_head_values(model, tokens: torch.Tensor) -> torch.Tensor:
    """Return the four post-attention, pre-W_O values at the ANS position."""
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    resid = model.tok_embed(tokens) + model.pos_embed(positions)
    layer = model.layers[0]
    causal_mask = torch.tril(
        torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device)
    ).unsqueeze(0)
    values = []
    for head in layer.heads:
        head_values, _ = head(resid, causal_mask)
        values.append(head_values[:, 10, :])
    ans_values = torch.stack(values, dim=1)
    expected_shape = (tokens.shape[0], model.n_heads, layer.d_head)
    if ans_values.shape != expected_shape:
        raise AssertionError(f"unexpected pre-W_O head-value shape: {ans_values.shape}")
    return ans_values


def projection_metrics(components: torch.Tensor, pca: dict) -> dict[str, torch.Tensor]:
    directions = pca["directions"]
    centered_unembedding = pca["centered"]
    rank = pca["rank"]
    top3 = directions[:3]

    coordinates = components @ top3.T
    total_energy = components.square().sum(dim=-1)
    top3_energy = coordinates.square().sum(dim=-1)
    top3_total_fraction = top3_energy / total_energy

    span_coordinates = components @ directions[:rank].T
    span_energy = span_coordinates.square().sum(dim=-1)
    top3_span_fraction = top3_energy / span_energy

    centered_logits = components @ centered_unembedding.T
    centered_unembedding_scores = centered_unembedding @ top3.T
    reconstructed_centered_logits = coordinates @ centered_unembedding_scores.T
    centered_logit_energy = centered_logits.square().sum(dim=-1)
    reconstructed_logit_energy = reconstructed_centered_logits.square().sum(dim=-1)
    top3_logit_fraction = reconstructed_logit_energy / centered_logit_energy

    fractions = torch.stack(
        [top3_total_fraction, top3_span_fraction, top3_logit_fraction], dim=-1
    )
    if not bool(torch.isfinite(fractions).all()):
        raise AssertionError("projection metrics contain non-finite values")
    if bool((fractions < -1e-6).any()) or bool((fractions > 1.0 + 1e-5).any()):
        raise AssertionError(f"projection fraction outside [0, 1]: {fractions.min()}, {fractions.max()}")

    return {
        "coordinates": coordinates,
        "norm": components.norm(dim=-1),
        "top3_total_vector_energy_fraction": top3_total_fraction,
        "top3_centered_unembedding_span_energy_fraction": top3_span_fraction,
        "top3_centered_logit_effect_energy_fraction": top3_logit_fraction,
    }


def global_head_sum_projection_accuracy(model, pcas: dict[str, dict]) -> dict:
    device = next(model.parameters()).device
    all_nums = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)])
    labels = all_nums.max(dim=1).values
    vocab_sizes = {"full_vocabulary": 14, "digits_only": 10}
    correct_baseline = {basis_name: 0 for basis_name in pcas}
    correct = {basis_name: {k: 0 for k in (1, 2, 3)} for basis_name in pcas}
    prediction_counts = {
        basis_name: {
            k: torch.zeros(vocab_sizes[basis_name], dtype=torch.long) for k in (1, 2, 3)
        }
        for basis_name in pcas
    }
    special_prediction_counts = {k: 0 for k in (1, 2, 3)}
    max_abs_equivalence_error = {
        basis_name: {k: 0.0 for k in (1, 2, 3)} for basis_name in pcas
    }
    unembedding = model.unembed.weight.detach()
    prepared = {}
    for basis_name, pca in pcas.items():
        centered_unembedding = pca["centered"].to(device)
        directions = pca["directions"].to(device)
        prepared[basis_name] = {}
        for k in (1, 2, 3):
            basis = directions[:k]
            low_dim_unembedding = centered_unembedding @ basis.T
            prepared[basis_name][k] = {
                "basis": basis,
                "centered_unembedding": centered_unembedding,
                "low_dim_unembedding": low_dim_unembedding,
            }

    for start in range(0, len(all_nums), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(all_nums))
        nums = all_nums[start:end].to(device)
        batch_labels = labels[start:end].to(device)
        components = extract_component_vectors(model, tokenize(nums))
        head_sum = components[:, 4]

        for basis_name, vocab_size in vocab_sizes.items():
            baseline_prediction = (head_sum @ unembedding[:vocab_size].T).argmax(dim=1)
            correct_baseline[basis_name] += int((baseline_prediction == batch_labels).sum())

        for basis_name in pcas:
            for k in (1, 2, 3):
                row = prepared[basis_name][k]
                basis = row["basis"]
                low_dim_unembedding = row["low_dim_unembedding"]
                head_coordinates = head_sum @ basis.T
                direct_logits = head_coordinates @ low_dim_unembedding.T

                projected_head_sum = head_coordinates @ basis
                reference_logits = projected_head_sum @ row["centered_unembedding"].T
                error = float((direct_logits - reference_logits).abs().max())
                max_abs_equivalence_error[basis_name][k] = max(
                    max_abs_equivalence_error[basis_name][k], error
                )
                if not torch.allclose(direct_logits, reference_logits, atol=2e-4, rtol=1e-5):
                    raise AssertionError(
                        f"direct low-dimensional logits disagree for {basis_name}, k={k}: {error}"
                    )

                prediction = direct_logits.argmax(dim=1)
                correct[basis_name][k] += int((prediction == batch_labels).sum())
                prediction_counts[basis_name][k] += torch.bincount(
                    prediction.cpu(), minlength=vocab_sizes[basis_name]
                )
                if basis_name == "full_vocabulary":
                    special_prediction_counts[k] += int((prediction >= 10).sum())

    total = len(all_nums)
    for basis_name, count in correct_baseline.items():
        if count != total:
            raise AssertionError(f"{basis_name} head-sum baseline is not perfect: {count}/{total}")

    return {
        "description": (
            "The summed post-W_O head output is projected to k unembedding PCs, "
            "then logits are computed directly as (batch x k) @ (k x vocab). "
            "No 64d vector is reconstructed for the evaluated logits."
        ),
        "formula": {
            "head_coordinates": "z = head_sum @ P_k.T",
            "low_dim_unembedding": "U_low = (W_U - mean(W_U)) @ P_k.T",
            "relative_logits": "z @ U_low.T",
        },
        "n_inputs": total,
        "baseline_head_sum_only": {
            basis_name: {
                "accuracy": correct_baseline[basis_name] / total,
                "correct": correct_baseline[basis_name],
                "total": total,
                "argmax_size": vocab_sizes[basis_name],
            }
            for basis_name in pcas
        },
        "by_unembedding_basis": {
            basis_name: {
                str(k): {
                    "accuracy": correct[basis_name][k] / total,
                    "correct": correct[basis_name][k],
                    "total": total,
                    "argmax_size": vocab_sizes[basis_name],
                    "head_coordinate_shape_per_input": [k],
                    "low_dim_unembedding_transposed_shape": [k, vocab_sizes[basis_name]],
                    "low_dim_unembedding_transposed": [
                        [float(value) for value in row]
                        for row in prepared[basis_name][k]["low_dim_unembedding"].T.cpu()
                    ],
                    "max_abs_error_vs_centered_projected_64d_logits": (
                        max_abs_equivalence_error[basis_name][k]
                    ),
                    "special_token_prediction_count": (
                        special_prediction_counts[k] if basis_name == "full_vocabulary" else None
                    ),
                    "prediction_distribution": {
                        str(token): int(prediction_counts[basis_name][k][token])
                        for token in range(vocab_sizes[basis_name])
                        if int(prediction_counts[basis_name][k][token]) > 0
                    },
                }
                for k in (1, 2, 3)
            }
            for basis_name in pcas
        },
    }


def plot_direct_readout_accuracy(global_accuracy: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    xs = [1, 2, 3]
    styles = [
        ("full_vocabulary", "Full vocabulary: 14-way argmax", "#2563eb", "o"),
        ("digits_only", "Digits only: 10-way argmax", "#dc2626", "s"),
    ]
    for basis_name, label, color, marker in styles:
        ys = [
            global_accuracy["by_unembedding_basis"][basis_name][str(k)]["accuracy"]
            for k in xs
        ]
        ax.plot(xs, ys, color=color, marker=marker, linewidth=2.0, markersize=7, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.3f}",
                xy=(x, y),
                xytext=(0, 9 if basis_name == "full_vocabulary" else -15),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
            )
    ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xlim(0.8, 3.2)
    ax.set_ylim(0.35, 1.04)
    ax.set_xlabel("Unembedding PCs used in the direct final readout")
    ax.set_ylabel("Max-of-list accuracy over all 100,000 inputs")
    ax.set_title("Model 1: direct low-dimensional unembedding readout")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    READOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(READOUT_OUT, dpi=180, facecolor="white")
    plt.close(fig)


def global_per_head_lowdim_output_accuracy(model, pcas: dict[str, dict]) -> dict:
    device = next(model.parameters()).device
    all_nums = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)])
    labels = all_nums.max(dim=1).values
    vocab_sizes = {"full_vocabulary": 14, "digits_only": 10}
    correct = {basis_name: {k: 0 for k in (1, 2, 3)} for basis_name in pcas}
    prediction_counts = {
        basis_name: {
            k: torch.zeros(vocab_sizes[basis_name], dtype=torch.long) for k in (1, 2, 3)
        }
        for basis_name in pcas
    }
    special_prediction_counts = {k: 0 for k in (1, 2, 3)}
    max_abs_coordinate_error = {
        basis_name: {k: 0.0 for k in (1, 2, 3)} for basis_name in pcas
    }

    layer = model.layers[0]
    w_o = layer.W_O.weight.detach()
    output_matrices = torch.stack(
        [
            w_o[:, head_idx * layer.d_head : (head_idx + 1) * layer.d_head].T
            for head_idx in range(model.n_heads)
        ],
        dim=0,
    )
    if output_matrices.shape != (4, 16, 64):
        raise AssertionError(f"unexpected per-head output-matrix shape: {output_matrices.shape}")

    prepared = {}
    for basis_name, pca in pcas.items():
        centered_unembedding = pca["centered"].to(device)
        directions = pca["directions"].to(device)
        prepared[basis_name] = {}
        for k in (1, 2, 3):
            basis = directions[:k]
            reduced_output_matrices = output_matrices @ basis.T
            low_dim_unembedding = centered_unembedding @ basis.T
            prepared[basis_name][k] = {
                "basis": basis,
                "reduced_output_matrices": reduced_output_matrices,
                "low_dim_unembedding": low_dim_unembedding,
            }

    for start in range(0, len(all_nums), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(all_nums))
        nums = all_nums[start:end].to(device)
        batch_labels = labels[start:end].to(device)
        head_values = extract_ans_head_values(model, tokenize(nums))
        full_head_writes = torch.einsum("bhd,hdr->bhr", head_values, output_matrices)
        full_head_sum = full_head_writes.sum(dim=1)

        for basis_name in pcas:
            for k in (1, 2, 3):
                row = prepared[basis_name][k]
                low_head_writes = torch.einsum(
                    "bhd,hdk->bhk", head_values, row["reduced_output_matrices"]
                )
                low_head_sum = low_head_writes.sum(dim=1)

                reference_coordinates = full_head_sum @ row["basis"].T
                error = float((low_head_sum - reference_coordinates).abs().max())
                max_abs_coordinate_error[basis_name][k] = max(
                    max_abs_coordinate_error[basis_name][k], error
                )
                if not torch.allclose(low_head_sum, reference_coordinates, atol=2e-4, rtol=1e-5):
                    raise AssertionError(
                        f"per-head low-dimensional coordinates disagree for "
                        f"{basis_name}, k={k}: {error}"
                    )

                logits = low_head_sum @ row["low_dim_unembedding"].T
                prediction = logits.argmax(dim=1)
                correct[basis_name][k] += int((prediction == batch_labels).sum())
                prediction_counts[basis_name][k] += torch.bincount(
                    prediction.cpu(), minlength=vocab_sizes[basis_name]
                )
                if basis_name == "full_vocabulary":
                    special_prediction_counts[k] += int((prediction >= 10).sum())

    total = len(all_nums)
    return {
        "description": (
            "Each post-attention pre-W_O head value is mapped directly from 16d to k "
            "unembedding-PC coordinates with O_h_low = O_h @ P_k, then the four "
            "low-dimensional head writes are summed and unembedded in k dimensions."
        ),
        "formula": {
            "reduced_output_matrix": "O_h_low = O_h @ P_k",
            "head_write": "z_h = value_h @ O_h_low",
            "summed_write": "z = sum_h z_h",
            "relative_logits": "z @ U_low.T",
        },
        "n_inputs": total,
        "original_per_head_output_matrix_shape": [16, 64],
        "by_unembedding_basis": {
            basis_name: {
                str(k): {
                    "accuracy": correct[basis_name][k] / total,
                    "correct": correct[basis_name][k],
                    "total": total,
                    "argmax_size": vocab_sizes[basis_name],
                    "pre_W_O_head_value_shape_per_input": [4, 16],
                    "reduced_output_matrix_shape_per_head": [16, k],
                    "summed_head_coordinate_shape_per_input": [k],
                    "reduced_output_matrices": {
                        f"H{head_idx}": [
                            [float(value) for value in row]
                            for row in prepared[basis_name][k]["reduced_output_matrices"][
                                head_idx
                            ].cpu()
                        ]
                        for head_idx in range(model.n_heads)
                    },
                    "max_abs_coordinate_error_vs_64d_route": (
                        max_abs_coordinate_error[basis_name][k]
                    ),
                    "special_token_prediction_count": (
                        special_prediction_counts[k] if basis_name == "full_vocabulary" else None
                    ),
                    "prediction_distribution": {
                        str(token): int(prediction_counts[basis_name][k][token])
                        for token in range(vocab_sizes[basis_name])
                        if int(prediction_counts[basis_name][k][token]) > 0
                    },
                }
                for k in (1, 2, 3)
            }
            for basis_name in pcas
        },
    }


def plot_per_head_lowdim_output_accuracy(per_head_accuracy: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    xs = [1, 2, 3]
    styles = [
        ("full_vocabulary", "Four 16 x k outputs, 14-way argmax", "#2563eb", "o"),
        ("digits_only", "Four 16 x k outputs, 10-way argmax", "#dc2626", "s"),
    ]
    for basis_name, label, color, marker in styles:
        ys = [
            per_head_accuracy["by_unembedding_basis"][basis_name][str(k)]["accuracy"]
            for k in xs
        ]
        ax.plot(xs, ys, color=color, marker=marker, linewidth=2.0, markersize=7, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.3f}",
                xy=(x, y),
                xytext=(0, 9 if basis_name == "full_vocabulary" else -15),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
            )
    ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xlim(0.8, 3.2)
    ax.set_ylim(0.35, 1.04)
    ax.set_xlabel("Dimensions retained in each head's reduced output matrix")
    ax.set_ylabel("Max-of-list accuracy over all 100,000 inputs")
    ax.set_title("Model 1: per-head 16d to low-dimensional output path")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    PER_HEAD_READOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PER_HEAD_READOUT_OUT, dpi=180, facecolor="white")
    plt.close(fig)


def component_records(metrics: dict[str, torch.Tensor], example_idx: int) -> dict:
    records = {}
    for component_idx, label in enumerate(COMPONENT_LABELS):
        records[label] = {
            "pc_coordinates": [
                float(value) for value in metrics["coordinates"][example_idx, component_idx]
            ],
            "vector_norm": float(metrics["norm"][example_idx, component_idx]),
            "top3_total_vector_energy_fraction": float(
                metrics["top3_total_vector_energy_fraction"][example_idx, component_idx]
            ),
            "top3_centered_unembedding_span_energy_fraction": float(
                metrics["top3_centered_unembedding_span_energy_fraction"][example_idx, component_idx]
            ),
            "top3_centered_logit_effect_energy_fraction": float(
                metrics["top3_centered_logit_effect_energy_fraction"][example_idx, component_idx]
            ),
        }
    return records


def shared_limits(coordinates: torch.Tensor) -> list[tuple[float, float]]:
    limits = []
    for pc_idx in range(3):
        values = coordinates[..., pc_idx]
        low = min(float(values.min()), 0.0)
        high = max(float(values.max()), 0.0)
        span = max(high - low, 1.0)
        limits.append((low - 0.08 * span, high + 0.08 * span))
    return limits


def plot_projection_grid(
    examples: torch.Tensor,
    basis_results: dict[str, dict],
) -> None:
    basis_order = ["full_vocabulary", "digits_only"]
    basis_titles = {
        "full_vocabulary": "Full-vocabulary unembedding PCs",
        "digits_only": "Digit-only unembedding PCs",
    }
    limits = {
        basis_name: shared_limits(basis_results[basis_name]["metrics"]["coordinates"])
        for basis_name in basis_order
    }

    fig = plt.figure(figsize=(16.0, 25.0), constrained_layout=True)
    for example_idx, nums in enumerate(examples.tolist()):
        true_max = max(nums)
        for basis_idx, basis_name in enumerate(basis_order):
            ax = fig.add_subplot(
                len(examples), len(basis_order), example_idx * 2 + basis_idx + 1, projection="3d"
            )
            pca = basis_results[basis_name]["pca"]
            metrics = basis_results[basis_name]["metrics"]
            coords = metrics["coordinates"][example_idx].numpy()

            for component_idx, label in enumerate(COMPONENT_LABELS):
                marker = "*" if label == "sum" else "x"
                size = 125 if label == "sum" else 72
                linewidth = 0.9 if label == "sum" else 2.2
                ax.plot(
                    [0.0, coords[component_idx, 0]],
                    [0.0, coords[component_idx, 1]],
                    [0.0, coords[component_idx, 2]],
                    color=COMPONENT_COLORS[component_idx],
                    linewidth=0.8,
                    alpha=0.22,
                )
                ax.scatter(
                    coords[component_idx, 0],
                    coords[component_idx, 1],
                    coords[component_idx, 2],
                    color=COMPONENT_COLORS[component_idx],
                    marker=marker,
                    s=size,
                    linewidth=linewidth,
                    depthshade=False,
                    label=label,
                )

            ax.scatter(0.0, 0.0, 0.0, color="#6b7280", marker="o", s=18, depthshade=False)
            ax.set_xlim(limits[basis_name][0])
            ax.set_ylim(limits[basis_name][1])
            ax.set_zlim(limits[basis_name][2])
            ax.set_xlabel("PC1", labelpad=7)
            ax.set_ylabel("PC2", labelpad=7)
            ax.set_zlabel("PC3", labelpad=7)
            explained = pca["explained"]
            ax.set_title(
                f"{basis_titles[basis_name]} "
                f"({float(explained[0]):.1%}, {float(explained[1]):.1%}, {float(explained[2]):.1%})\n"
                f"input {nums}, true max {true_max}",
                pad=13,
                fontsize=10,
            )
            sum_logit_fraction = float(
                metrics["top3_centered_logit_effect_energy_fraction"][example_idx, 4]
            )
            ax.text2D(
                0.02,
                0.02,
                f"sum centered-logit energy in top 3: {sum_logit_fraction:.3%}",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.view_init(elev=22, azim=-58)
            ax.set_proj_type("ortho")
            ax.set_box_aspect((1.15, 1.0, 0.85))
            ax.grid(alpha=0.2)
            if example_idx == 0:
                ax.legend(loc="upper left", frameon=False, fontsize=8)

    fig.suptitle(
        "Model 1: actual per-head ANS writes in unembedding top-3 PC spaces",
        fontsize=16,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)


def make_overlay_data(basis_results: dict[str, dict]) -> dict:
    overlay = {}
    for basis_name, result in basis_results.items():
        pca = result["pca"]
        head_coordinates = result["metrics"]["coordinates"]
        digit_coordinates = pca["centered"][:10] @ pca["directions"][:3].T

        max_digit_radius = float(digit_coordinates.norm(dim=-1).max())
        max_head_radius = float(head_coordinates.norm(dim=-1).max())
        head_display_scale = 0.90 * max_digit_radius / max_head_radius

        sample_alignment = []
        for example_idx, true_max in enumerate(SELECTED_MAXIMA):
            head_sum = head_coordinates[example_idx, 4]
            dot_products = head_sum @ digit_coordinates.T
            cosine = dot_products / (
                head_sum.norm() * digit_coordinates.norm(dim=-1)
            )
            sample_alignment.append(
                {
                    "true_max": true_max,
                    "top3_dot_product_prediction": int(dot_products.argmax()),
                    "true_max_cosine": float(cosine[true_max]),
                    "nearest_digit_by_cosine": int(cosine.argmax()),
                    "nearest_digit_cosine": float(cosine.max()),
                    "cosine_by_digit": [float(value) for value in cosine],
                    "centered_dot_product_by_digit": [float(value) for value in dot_products],
                }
            )

        overlay[basis_name] = {
            "head_display_scale": head_display_scale,
            "max_digit_top3_radius": max_digit_radius,
            "max_raw_head_top3_radius": max_head_radius,
            "digit_pc_coordinates": [
                [float(value) for value in row] for row in digit_coordinates
            ],
            "sample_alignment": sample_alignment,
        }
    return overlay


def plot_unembedding_overlay(
    examples: torch.Tensor,
    basis_results: dict[str, dict],
    overlay_data: dict,
) -> None:
    basis_order = ["full_vocabulary", "digits_only"]
    basis_titles = {
        "full_vocabulary": "Full-vocabulary unembedding PCs",
        "digits_only": "Digit-only unembedding PCs",
    }
    display_coordinates = {}
    digit_coordinates = {}
    limits = {}
    for basis_name in basis_order:
        display_scale = overlay_data[basis_name]["head_display_scale"]
        display_coordinates[basis_name] = (
            basis_results[basis_name]["metrics"]["coordinates"] * display_scale
        )
        digit_coordinates[basis_name] = torch.tensor(
            overlay_data[basis_name]["digit_pc_coordinates"]
        )
        all_coordinates = torch.cat(
            [
                digit_coordinates[basis_name],
                display_coordinates[basis_name].reshape(-1, 3),
                torch.zeros(1, 3),
            ],
            dim=0,
        )
        limits[basis_name] = shared_limits(all_coordinates)

    fig = plt.figure(figsize=(16.0, 25.0), constrained_layout=True)
    for example_idx, nums in enumerate(examples.tolist()):
        true_max = max(nums)
        for basis_idx, basis_name in enumerate(basis_order):
            ax = fig.add_subplot(
                len(examples), len(basis_order), example_idx * 2 + basis_idx + 1, projection="3d"
            )
            digits = digit_coordinates[basis_name].numpy()
            heads = display_coordinates[basis_name][example_idx].numpy()
            display_scale = overlay_data[basis_name]["head_display_scale"]
            alignment = overlay_data[basis_name]["sample_alignment"][example_idx]

            ax.plot(
                digits[:, 0],
                digits[:, 1],
                digits[:, 2],
                color="#6b7280",
                linewidth=1.0,
                alpha=0.65,
                label="digit order",
            )
            ax.scatter(
                digits[:, 0],
                digits[:, 1],
                digits[:, 2],
                c=np.arange(10),
                cmap="viridis",
                s=52,
                edgecolor="white",
                linewidth=0.6,
                depthshade=False,
                label="centered digit unembeddings",
            )
            ax.scatter(
                digits[true_max, 0],
                digits[true_max, 1],
                digits[true_max, 2],
                facecolors="none",
                edgecolors="#111827",
                marker="o",
                s=145,
                linewidth=1.5,
                depthshade=False,
                label="true max digit",
            )

            digit_spans = np.maximum(np.ptp(digits, axis=0), 1e-6)
            for digit in range(10):
                ax.text(
                    digits[digit, 0] + 0.018 * digit_spans[0],
                    digits[digit, 1],
                    digits[digit, 2] + 0.018 * digit_spans[2],
                    str(digit),
                    fontsize=7,
                )

            for component_idx, label in enumerate(COMPONENT_LABELS):
                marker = "*" if label == "sum" else "x"
                size = 130 if label == "sum" else 72
                linewidth = 0.9 if label == "sum" else 2.2
                ax.plot(
                    [0.0, heads[component_idx, 0]],
                    [0.0, heads[component_idx, 1]],
                    [0.0, heads[component_idx, 2]],
                    color=COMPONENT_COLORS[component_idx],
                    linewidth=0.8,
                    alpha=0.25,
                )
                ax.scatter(
                    heads[component_idx, 0],
                    heads[component_idx, 1],
                    heads[component_idx, 2],
                    color=COMPONENT_COLORS[component_idx],
                    marker=marker,
                    s=size,
                    linewidth=linewidth,
                    depthshade=False,
                    label=label,
                )

            ax.scatter(0.0, 0.0, 0.0, color="#6b7280", marker="o", s=16, depthshade=False)
            ax.set_xlim(limits[basis_name][0])
            ax.set_ylim(limits[basis_name][1])
            ax.set_zlim(limits[basis_name][2])
            ax.set_xlabel("PC1", labelpad=7)
            ax.set_ylabel("PC2", labelpad=7)
            ax.set_zlabel("PC3", labelpad=7)
            ax.set_title(
                f"{basis_titles[basis_name]} | heads x {display_scale:.5f}\n"
                f"input {nums}, true max {true_max}",
                pad=13,
                fontsize=10,
            )
            ax.text2D(
                0.02,
                0.02,
                f"top-3 dot-product pred: {alignment['top3_dot_product_prediction']} | "
                f"cos(sum, centered U[{true_max}]): {alignment['true_max_cosine']:+.3f}",
                transform=ax.transAxes,
                fontsize=8,
            )
            ax.view_init(elev=22, azim=-58)
            ax.set_proj_type("ortho")
            ax.set_box_aspect((1.15, 1.0, 0.85))
            ax.grid(alpha=0.2)
            if example_idx == 0:
                ax.legend(loc="upper left", frameon=False, fontsize=7)

    fig.suptitle(
        "Model 1: head writes overlaid with centered digit unembeddings in top-3 PC space",
        fontsize=16,
    )
    OVERLAY_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OVERLAY_OUT, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model()
    model = model.to(device)

    examples = select_examples().to(device)
    tokens = tokenize(examples)
    with torch.no_grad():
        components = extract_component_vectors(model, tokens).cpu()

    unembedding = model.unembed.weight.detach().cpu().float()
    pcas = {
        "full_vocabulary": fit_pca(unembedding),
        "digits_only": fit_pca(unembedding[:10]),
    }
    if pcas["full_vocabulary"]["rank"] != 13 or pcas["digits_only"]["rank"] != 9:
        raise AssertionError("unexpected centered unembedding rank")

    basis_results = {}
    for basis_name, pca in pcas.items():
        basis_results[basis_name] = {
            "pca": pca,
            "metrics": projection_metrics(components, pca),
        }

    overlay_data = make_overlay_data(basis_results)

    with torch.no_grad():
        global_accuracy = global_head_sum_projection_accuracy(model, pcas)
        per_head_lowdim_accuracy = global_per_head_lowdim_output_accuracy(model, pcas)

    for basis_name in pcas:
        for k in (1, 2, 3):
            direct_row = global_accuracy["by_unembedding_basis"][basis_name][str(k)]
            per_head_row = per_head_lowdim_accuracy["by_unembedding_basis"][basis_name][str(k)]
            if direct_row["correct"] != per_head_row["correct"]:
                raise AssertionError(
                    f"per-head and sum-first accuracy differ for {basis_name}, k={k}"
                )
            if direct_row["prediction_distribution"] != per_head_row["prediction_distribution"]:
                raise AssertionError(
                    f"per-head and sum-first predictions differ for {basis_name}, k={k}"
                )

    digit_unembedding = unembedding[:10]
    baseline_sample_logits = components[:, 4] @ digit_unembedding.T
    baseline_sample_predictions = baseline_sample_logits.argmax(dim=1)
    expected_labels = examples.cpu().max(dim=1).values
    if not torch.equal(baseline_sample_predictions, expected_labels):
        raise AssertionError("head-sum baseline failed on a selected example")

    samples = []
    for example_idx, nums in enumerate(examples.cpu().tolist()):
        sample = {
            "numbers": nums,
            "true_max": max(nums),
            "tokens": tokenize(torch.tensor([nums])).squeeze(0).tolist(),
            "baseline_head_sum_prediction": int(baseline_sample_predictions[example_idx]),
            "bases": {},
        }
        for basis_name, result in basis_results.items():
            top3 = result["pca"]["directions"][:3]
            centered_unembedding = result["pca"]["centered"]
            head_coordinates = components[example_idx, 4] @ top3.T
            low_dim_unembedding = centered_unembedding @ top3.T
            direct_low_dim_logits = head_coordinates @ low_dim_unembedding.T
            projected_prediction = int(direct_low_dim_logits.argmax())
            sample["bases"][basis_name] = {
                "top3_direct_low_dim_prediction": projected_prediction,
                "top3_direct_low_dim_logits": [float(value) for value in direct_low_dim_logits],
                "components": component_records(result["metrics"], example_idx),
            }
        samples.append(sample)

    result = {
        "description": (
            "Actual layer-0 ANS-position outputs H0..H3 after each head's W_O slice, "
            "plus their sum, projected without mean subtraction into the top three "
            "principal directions of either the centered 14-token unembedding or the "
            "centered digit-only unembedding."
        ),
        "hf_repo": "andyrdt/04_2026_puzzle_1a",
        "model_config": config,
        "seed": SEED,
        "selected_maxima": SELECTED_MAXIMA,
        "component_shape": list(components.shape),
        "projection_note": (
            "Head writes are vectors from the residual-stream origin, so coordinates "
            "are h @ PC.T; the unembedding PCA mean is not subtracted from h."
        ),
        "basis_metadata": {
            basis_name: {
                "token_vector_shape": list(pca["centered"].shape),
                "centered_rank": pca["rank"],
                "top3_explained_variance": [float(value) for value in pca["explained"][:3]],
                "top3_cumulative_explained_variance": float(pca["explained"][:3].sum()),
            }
            for basis_name, pca in pcas.items()
        },
        "metric_definitions": {
            "top3_total_vector_energy_fraction": "||h P3^T||^2 / ||h||^2",
            "top3_centered_unembedding_span_energy_fraction": (
                "||h P3^T||^2 / ||h Prank^T||^2"
            ),
            "top3_centered_logit_effect_energy_fraction": (
                "||top3 reconstruction of h @ centered_W_U^T||^2 / "
                "||h @ centered_W_U^T||^2"
            ),
        },
        "samples": samples,
        "overlay_display": overlay_data,
        "global_head_sum_projection_accuracy": global_accuracy,
        "per_head_low_dimensional_output_accuracy": per_head_lowdim_accuracy,
    }

    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(encoded)
    plot_projection_grid(examples.cpu(), basis_results)
    plot_unembedding_overlay(examples.cpu(), basis_results, overlay_data)
    plot_direct_readout_accuracy(global_accuracy)
    plot_per_head_lowdim_output_accuracy(per_head_lowdim_accuracy)

    print("basis,k,accuracy,correct,total")
    for basis_name, rows in global_accuracy["by_unembedding_basis"].items():
        for k, row in rows.items():
            print(
                f"{basis_name},{k},{row['accuracy']:.6f},{row['correct']},{row['total']},"
                f"specials={row['special_token_prediction_count']}"
            )
    print("sample,basis,component,total_energy_fraction,span_energy_fraction,logit_effect_fraction")
    for sample in samples:
        for basis_name, basis in sample["bases"].items():
            for component_name, row in basis["components"].items():
                print(
                    f"{sample['numbers']},{basis_name},{component_name},"
                    f"{row['top3_total_vector_energy_fraction']:.6f},"
                    f"{row['top3_centered_unembedding_span_energy_fraction']:.6f},"
                    f"{row['top3_centered_logit_effect_energy_fraction']:.6f}"
                )
    print(f"wrote,{OUT}")
    print(f"wrote,{OVERLAY_OUT}")
    print(f"wrote,{READOUT_OUT}")
    print(f"wrote,{PER_HEAD_READOUT_OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
