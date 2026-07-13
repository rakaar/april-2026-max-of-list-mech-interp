#!/usr/bin/env python3
"""Analyze unit-unembedding retrains against the released Puzzle 1a model."""

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
MODEL_PY = ROOT / "04_2026" / "model.py"
CHECKPOINT_ROOT = ROOT / "04_2026" / "puzzle1a" / "checkpoints"
JSON_OUT = ROOT / "docs" / "assets" / "model1_unit_unembed_experiment.json"
SUMMARY_PNG = ROOT / "docs" / "assets" / "model1_unit_unembed_summary.png"
LOWDIM_PNG = ROOT / "docs" / "assets" / "model1_unit_unembed_lowdim.png"
RECRUITMENT_PNG = ROOT / "docs" / "assets" / "model1_unit_unembed_recruitment.png"
ANGULAR_PNG = ROOT / "docs" / "assets" / "model1_unit_unembed_angular.png"
HF_REPO = "andyrdt/04_2026_puzzle_1a"
NUMBER_POSITIONS = torch.tensor([1, 3, 5, 7, 9])
EXPECTED_COUNTS = torch.tensor(
    [1, 31, 211, 781, 2101, 4651, 9031, 15961, 26281, 40951]
)
BATCH_SIZE = 4096
PRIMARY_MODEL = "unit_seed42"
MODEL_ORDER = ("released_original", "unit_seed42", "unit_seed43", "unit_seed44")
MODEL_LABELS = {
    "released_original": "released",
    "unit_seed42": "unit s42",
    "unit_seed43": "unit s43",
    "unit_seed44": "unit s44",
}
HEAD_COLORS = ("#2563eb", "#f97316", "#16a34a", "#dc2626")


def load_model_module():
    spec = importlib.util.spec_from_file_location("unit_unembed_model", MODEL_PY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_model(name: str, device: torch.device):
    module = load_model_module()
    if name == "released_original":
        config_path = Path(hf_hub_download(HF_REPO, "config.json"))
        weights_path = Path(hf_hub_download(HF_REPO, "model.pt"))
    else:
        seed = int(name.removeprefix("unit_seed"))
        checkpoint = CHECKPOINT_ROOT / f"unit_unembed_seed{seed}"
        config_path = checkpoint / "config.json"
        weights_path = checkpoint / "model.pt"
    config = json.loads(config_path.read_text())
    model = module.AttentionOnlyTransformer.from_config(config["model"])
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    model.eval().to(device)
    return model, config


def all_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    numbers = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)])
    labels = numbers.max(dim=1).values
    counts = torch.bincount(labels, minlength=10)
    if not torch.equal(counts, EXPECTED_COUNTS):
        raise AssertionError(f"unexpected true-max counts: {counts.tolist()}")
    return numbers, labels


def tokenize(numbers: torch.Tensor) -> torch.Tensor:
    tokens = torch.empty(
        (numbers.shape[0], 11), dtype=torch.long, device=numbers.device
    )
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


def output_matrices(model) -> torch.Tensor:
    stored = model.layers[0].W_O.weight.detach()
    d_head = model.layers[0].d_head
    return torch.stack(
        [
            stored[:, head * d_head : (head + 1) * d_head].T
            for head in range(model.layers[0].n_heads)
        ]
    )


def extract_components(model, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    seq_len = tokens.shape[1]
    positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
    residual = model.tok_embed(tokens) + model.pos_embed(positions)
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=tokens.device)
    ).unsqueeze(0)
    ans_values = []
    source_values = []
    attention_rows = []
    for head in model.layers[0].heads:
        output, attention = head(residual, causal_mask)
        ans_values.append(output[:, -1])
        source_values.append(head.W_V(residual))
        attention_rows.append(attention[:, -1])
    ans_values_t = torch.stack(ans_values, dim=1)
    source_values_t = torch.stack(source_values, dim=1)
    attention_rows_t = torch.stack(attention_rows, dim=1)
    matrices = output_matrices(model)
    head_writes = torch.einsum("bhd,hdr->bhr", ans_values_t, matrices)
    head_sum = head_writes.sum(dim=1)
    ans_residual = residual[:, -1]
    return {
        "residual": residual,
        "ans_residual": ans_residual,
        "ans_values": ans_values_t,
        "source_values": source_values_t,
        "attention": attention_rows_t,
        "output_matrices": matrices,
        "head_writes": head_writes,
        "head_sum": head_sum,
        "final_state": ans_residual + head_sum,
    }


def accuracy_record(correct: torch.Tensor, counts: torch.Tensor) -> dict:
    return {
        "correct": int(correct.sum()),
        "total": int(counts.sum()),
        "accuracy": float(correct.sum() / counts.sum()),
        "by_true_max": {
            str(maximum): {
                "correct": int(correct[maximum]),
                "total": int(counts[maximum]),
                "accuracy": float(correct[maximum] / counts[maximum]),
            }
            for maximum in range(10)
        },
    }


def scheme_name(scheme: int) -> str:
    sources = ["max" if scheme & (1 << head) else "ANS" for head in range(4)]
    return ", ".join(f"H{head}={source}" for head, source in enumerate(sources))


def analyze_core(model, numbers_all: torch.Tensor, labels_all: torch.Tensor) -> dict:
    device = next(model.parameters()).device
    counts = torch.bincount(labels_all, minlength=10)
    unit = model.unembed.weight.detach()
    norms = unit.norm(dim=1).cpu()

    variant_names = ("actual", "head_sum_only", "full64_cosine", "head_sum_cosine")
    correct = {name: torch.zeros(10, dtype=torch.long) for name in variant_names}
    special_predictions = {name: 0 for name in variant_names}
    margin_min = {"actual": float("inf"), "head_sum_only": float("inf")}

    attention_fields = ("max_mass", "ans_mass", "nonmax_number_mass", "other_mass")
    attention_sum = {
        field: torch.zeros((10, 4), dtype=torch.float64)
        for field in attention_fields
    }
    attention_sq_sum = {
        field: torch.zeros((10, 4), dtype=torch.float64)
        for field in attention_fields
    }
    max_mass_beats_ans = torch.zeros((10, 4), dtype=torch.long)
    top_source_counts = torch.zeros((10, 4, 4), dtype=torch.long)
    source_categories = ("ANS", "max_number", "nonmax_number", "BOS_or_SEP")

    intervention_names = ("force_ANS", "force_max", "zero")
    intervention_correct = {
        name: torch.zeros((4, 10), dtype=torch.long)
        for name in intervention_names
    }
    scheme_correct = torch.zeros((16, 10), dtype=torch.long)
    max_abs_forward_error = 0.0

    number_positions_device = NUMBER_POSITIONS.to(device)
    scheme_bits = torch.tensor(
        [[bool(scheme & (1 << head)) for head in range(4)] for scheme in range(16)],
        device=device,
    )

    with torch.no_grad():
        for start in range(0, len(numbers_all), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(numbers_all))
            numbers = numbers_all[start:end].to(device)
            labels = labels_all[start:end].to(device)
            tokens = tokenize(numbers)
            components = extract_components(model, tokens)
            final_state = components["final_state"]
            head_sum = components["head_sum"]
            unembedding = model.unembed.weight.detach()

            manual_logits = final_state @ unembedding.T
            if start == 0:
                model_logits, _ = model(tokens)
                error = float((manual_logits - model_logits[:, -1]).abs().max())
                max_abs_forward_error = max(max_abs_forward_error, error)
                if not torch.allclose(
                    manual_logits, model_logits[:, -1], rtol=1e-5, atol=2e-4
                ):
                    raise AssertionError(f"manual forward mismatch: {error}")

            states = {"actual": final_state, "head_sum_only": head_sum}
            predictions = {}
            for state_name, state in states.items():
                logits = state @ unembedding.T
                prediction = logits.argmax(dim=1)
                predictions[state_name] = prediction
                top2 = logits[:, :10].topk(2, dim=1).values
                margin_min[state_name] = min(
                    margin_min[state_name], float((top2[:, 0] - top2[:, 1]).min())
                )
                cosine = F.normalize(state, dim=1) @ F.normalize(
                    unembedding, dim=1
                ).T
                cosine_prediction = cosine.argmax(dim=1)
                cosine_name = (
                    "full64_cosine" if state_name == "actual" else "head_sum_cosine"
                )
                predictions[cosine_name] = cosine_prediction

            labels_cpu = labels.cpu()
            for name, prediction in predictions.items():
                prediction_cpu = prediction.cpu()
                special_predictions[name] += int((prediction_cpu >= 10).sum())
                for maximum in range(10):
                    selected = labels_cpu == maximum
                    correct[name][maximum] += int(
                        (prediction_cpu[selected] == labels_cpu[selected]).sum()
                    )

            attention = components["attention"]
            is_max_slot = numbers == labels[:, None]
            number_attention = attention[:, :, number_positions_device]
            max_mass = (
                number_attention * is_max_slot[:, None, :].to(number_attention.dtype)
            ).sum(dim=2)
            nonmax_number_mass = (
                number_attention * (~is_max_slot)[:, None, :].to(number_attention.dtype)
            ).sum(dim=2)
            ans_mass = attention[:, :, -1]
            other_mass = 1.0 - max_mass - nonmax_number_mass - ans_mass
            field_values = {
                "max_mass": max_mass,
                "ans_mass": ans_mass,
                "nonmax_number_mass": nonmax_number_mass,
                "other_mass": other_mass,
            }
            top_positions = attention.argmax(dim=2)
            top_categories = torch.full_like(top_positions, 3)
            top_categories[top_positions == 10] = 0
            for slot, position in enumerate(NUMBER_POSITIONS.tolist()):
                at_position = top_positions == position
                slot_is_max = is_max_slot[:, slot][:, None]
                top_categories[at_position & slot_is_max] = 1
                top_categories[at_position & ~slot_is_max] = 2

            for maximum in range(10):
                selected = labels == maximum
                selected_cpu = selected.cpu()
                if not bool(selected.any()):
                    continue
                for field, values in field_values.items():
                    chosen = values[selected].double().cpu()
                    attention_sum[field][maximum] += chosen.sum(dim=0)
                    attention_sq_sum[field][maximum] += chosen.square().sum(dim=0)
                max_mass_beats_ans[maximum] += (
                    max_mass[selected] > ans_mass[selected]
                ).sum(dim=0).cpu()
                categories = top_categories[selected].cpu()
                for head in range(4):
                    top_source_counts[maximum, head] += torch.bincount(
                        categories[:, head], minlength=4
                    )

            source_values = components["source_values"]
            matrices = components["output_matrices"]
            batch = len(numbers)
            batch_idx = torch.arange(batch, device=device)[:, None]
            head_idx = torch.arange(4, device=device)[None, :]
            masked_max_attention = number_attention.masked_fill(
                ~is_max_slot[:, None, :], float("-inf")
            )
            selected_slots = masked_max_attention.argmax(dim=2)
            selected_positions = number_positions_device[selected_slots]
            self_values = source_values[:, :, -1]
            max_values = source_values[batch_idx, head_idx, selected_positions]
            self_writes = torch.einsum("bhd,hdr->bhr", self_values, matrices)
            max_writes = torch.einsum("bhd,hdr->bhr", max_values, matrices)
            actual_writes = components["head_writes"]
            ans_residual = components["ans_residual"]

            for head in range(4):
                alternatives = {
                    "force_ANS": self_writes[:, head],
                    "force_max": max_writes[:, head],
                    "zero": torch.zeros_like(actual_writes[:, head]),
                }
                for intervention, replacement in alternatives.items():
                    state = (
                        ans_residual
                        + head_sum
                        - actual_writes[:, head]
                        + replacement
                    )
                    pred = (state @ unembedding.T).argmax(dim=1).cpu()
                    for maximum in range(10):
                        selected = labels_cpu == maximum
                        intervention_correct[intervention][head, maximum] += int(
                            (pred[selected] == labels_cpu[selected]).sum()
                        )

            onehot_writes = torch.where(
                scheme_bits[None, :, :, None],
                max_writes[:, None, :, :],
                self_writes[:, None, :, :],
            ).sum(dim=2)
            scheme_states = ans_residual[:, None, :] + onehot_writes
            scheme_logits = torch.einsum("bsr,vr->bsv", scheme_states, unembedding)
            scheme_predictions = scheme_logits.argmax(dim=2).cpu()
            for maximum in range(10):
                selected = labels_cpu == maximum
                scheme_correct[:, maximum] += (
                    scheme_predictions[selected] == maximum
                ).sum(dim=0)

    attention = {}
    counts_f = counts.double()
    for maximum in range(10):
        row = {}
        for head in range(4):
            head_row = {}
            for field in attention_fields:
                mean = attention_sum[field][maximum, head] / counts_f[maximum]
                if counts[maximum] > 1:
                    variance = (
                        attention_sq_sum[field][maximum, head]
                        - counts_f[maximum] * mean.square()
                    ) / (counts_f[maximum] - 1)
                    std = variance.clamp_min(0).sqrt()
                else:
                    std = torch.tensor(0.0)
                head_row[field] = {"mean": float(mean), "std": float(std)}
            head_row["fraction_max_mass_beats_ANS"] = float(
                max_mass_beats_ans[maximum, head] / counts[maximum]
            )
            head_row["top_source_distribution"] = {
                source_categories[category]: {
                    "count": int(top_source_counts[maximum, head, category]),
                    "fraction": float(
                        top_source_counts[maximum, head, category] / counts[maximum]
                    ),
                }
                for category in range(4)
            }
            row[f"H{head}"] = head_row
        attention[str(maximum)] = row

    interventions = {}
    for intervention in intervention_names:
        interventions[intervention] = {
            f"H{head}": accuracy_record(intervention_correct[intervention][head], counts)
            for head in range(4)
        }

    schemes = {}
    for maximum in range(10):
        total = int(counts[maximum])
        accuracies = scheme_correct[:, maximum].double() / total
        perfect = [scheme for scheme in range(16) if scheme_correct[scheme, maximum] == total]
        if perfect:
            minimum_reads = min(scheme.bit_count() for scheme in perfect)
            minimal = [scheme for scheme in perfect if scheme.bit_count() == minimum_reads]
        else:
            minimum_reads = None
            best = float(accuracies.max())
            minimal = [scheme for scheme in range(16) if float(accuracies[scheme]) == best]
        schemes[str(maximum)] = {
            "perfect_scheme_count": len(perfect),
            "minimum_max_reading_heads": minimum_reads,
            "minimal_perfect_schemes": [
                {"scheme": scheme, "description": scheme_name(scheme)}
                for scheme in minimal
            ] if perfect else [],
            "best_accuracy": float(accuracies.max()),
            "best_schemes_if_none_perfect": [
                {"scheme": scheme, "description": scheme_name(scheme)}
                for scheme in minimal
            ] if not perfect else [],
            "all_scheme_accuracies": {
                str(scheme): {
                    "description": scheme_name(scheme),
                    "correct": int(scheme_correct[scheme, maximum]),
                    "total": total,
                    "accuracy": float(accuracies[scheme]),
                }
                for scheme in range(16)
            },
        }

    return {
        "unembedding_row_norms": {
            "values": [float(value) for value in norms],
            "minimum": float(norms.min()),
            "maximum": float(norms.max()),
            "mean": float(norms.mean()),
            "std": float(norms.std(unbiased=False)),
            "max_abs_error_from_one": float((norms - 1.0).abs().max()),
        },
        "readout": {
            name: {
                **accuracy_record(correct[name], counts),
                "special_token_predictions": special_predictions[name],
            }
            for name in variant_names
        },
        "minimum_digit_margin": margin_min,
        "max_abs_manual_vs_model_logit_error": max_abs_forward_error,
        "attention_by_true_max": attention,
        "interventions": interventions,
        "onehot_ANS_or_max_schemes": schemes,
        "max_source_selection": (
            "For each example and head, force-max uses the maximum-valued number "
            "position receiving the largest actual attention among tied maxima."
        ),
    }


def fit_pca(rows: torch.Tensor) -> dict:
    rows = rows.detach().cpu().double()
    mean = rows.mean(dim=0, keepdim=True)
    centered = rows - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    explained = energy / energy.sum()
    rank = int(torch.linalg.matrix_rank(centered))
    return {
        "mean": mean.squeeze(0),
        "basis": vh.T,
        "singular_values": singular_values,
        "explained": explained,
        "rank": rank,
    }


def cumulative_predictions(
    states: torch.Tensor, candidates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contributions = states[:, :, None] * candidates.T[None, :, :]
    dot_scores = contributions.cumsum(dim=1)
    dot_predictions = dot_scores.argmax(dim=2)
    state_norms = states.square().cumsum(dim=1).sqrt().clamp_min(1e-12)
    candidate_norms = (
        candidates.square().cumsum(dim=1).sqrt().clamp_min(1e-12)
    )
    cosine_scores = dot_scores / (
        state_norms[:, :, None] * candidate_norms.T[None, :, :]
    )
    cosine_predictions = cosine_scores.argmax(dim=2)
    return dot_predictions, cosine_predictions, dot_scores


def serialize_curve(correct: torch.Tensor, total: int) -> dict:
    accuracies = correct.double() / total
    perfect = (correct == total).nonzero(as_tuple=False)
    return {
        "correct_by_k": [int(value) for value in correct],
        "accuracy_by_k": [float(value) for value in accuracies],
        "minimum_k_for_100_percent": (
            int(perfect[0, 0] + 1) if len(perfect) else None
        ),
    }


def analyze_lowdim(
    model, numbers_all: torch.Tensor, labels_all: torch.Tensor
) -> dict:
    device = next(model.parameters()).device
    unembedding_cpu = model.unembed.weight.detach().cpu().double()
    matrices_cpu = output_matrices(model).detach().cpu().double()
    bases = {
        "digit_unembedding": fit_pca(unembedding_cpu[:10]),
        "full_unembedding": fit_pca(unembedding_cpu),
        "output_matrix": fit_pca(matrices_cpu.reshape(64, 64)),
    }
    bases["digit_unembedding"]["max_k"] = bases["digit_unembedding"]["rank"]
    bases["full_unembedding"]["max_k"] = bases["full_unembedding"]["rank"]
    bases["output_matrix"]["max_k"] = bases["output_matrix"]["basis"].shape[1]

    states = ("actual", "head_sum_only")
    candidate_sets = ("digits", "full_vocabulary")
    readout_types = ("dot_raw", "cosine_raw", "cosine_centered")
    correct = {
        basis_name: {
            state: {
                candidate_set: {
                    readout: torch.zeros(basis["max_k"], dtype=torch.long)
                    for readout in readout_types
                }
                for candidate_set in candidate_sets
            }
            for state in states
        }
        for basis_name, basis in bases.items()
    }
    by_max_k3 = {
        basis_name: {
            state: {
                readout: torch.zeros((10, 10), dtype=torch.long)
                for readout in readout_types
            }
            for state in states
        }
        for basis_name in bases
    }
    centered_raw_candidate_offset_error = {basis_name: 0.0 for basis_name in bases}
    max_coordinate_error = {basis_name: 0.0 for basis_name in bases}

    with torch.no_grad():
        for start in range(0, len(numbers_all), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(numbers_all))
            numbers = numbers_all[start:end].to(device)
            labels = labels_all[start:end].to(device)
            labels_cpu = labels.cpu()
            components = extract_components(model, tokenize(numbers))
            state_vectors = {
                "actual": components["final_state"],
                "head_sum_only": components["head_sum"],
            }

            for basis_name, pca in bases.items():
                max_k = pca["max_k"]
                basis = pca["basis"][:, :max_k].to(
                    device=device, dtype=components["final_state"].dtype
                )
                raw_candidates = {
                    "digits": model.unembed.weight.detach()[:10] @ basis,
                    "full_vocabulary": model.unembed.weight.detach() @ basis,
                }
                centered_candidates = {
                    name: candidate
                    - candidate.mean(dim=0, keepdim=True)
                    for name, candidate in raw_candidates.items()
                }

                reduced_output = components["output_matrices"] @ basis
                routed_head_coordinates = torch.einsum(
                    "bhd,hdk->bhk", components["ans_values"], reduced_output
                ).sum(dim=1)
                reference_head_coordinates = components["head_sum"] @ basis
                error = float(
                    (routed_head_coordinates - reference_head_coordinates).abs().max()
                )
                max_coordinate_error[basis_name] = max(
                    max_coordinate_error[basis_name], error
                )
                if not torch.allclose(
                    routed_head_coordinates,
                    reference_head_coordinates,
                    rtol=1e-5,
                    atol=2e-4,
                ):
                    raise AssertionError(
                        f"per-head low-dimensional route failed for {basis_name}: {error}"
                    )

                for state_name, vector in state_vectors.items():
                    coordinates = vector @ basis
                    for candidate_name in candidate_sets:
                        raw = raw_candidates[candidate_name]
                        centered = centered_candidates[candidate_name]
                        dot_pred, cosine_raw_pred, _ = cumulative_predictions(
                            coordinates, raw
                        )
                        _, cosine_centered_pred, _ = cumulative_predictions(
                            coordinates, centered
                        )
                        candidate_offsets = raw - centered
                        offset_error = float(
                            (candidate_offsets - candidate_offsets[:1]).abs().max()
                        )
                        centered_raw_candidate_offset_error[basis_name] = max(
                            centered_raw_candidate_offset_error[basis_name],
                            offset_error,
                        )
                        predictions = {
                            "dot_raw": dot_pred,
                            "cosine_raw": cosine_raw_pred,
                            "cosine_centered": cosine_centered_pred,
                        }
                        for readout, prediction in predictions.items():
                            correct[basis_name][state_name][candidate_name][readout] += (
                                prediction.cpu() == labels_cpu[:, None]
                            ).sum(dim=0)

                        if candidate_name == "digits" and max_k >= 3:
                            for readout, prediction in predictions.items():
                                pred3 = prediction[:, 2].cpu()
                                for maximum in range(10):
                                    selected = labels_cpu == maximum
                                    by_max_k3[basis_name][state_name][readout][maximum] += (
                                        torch.bincount(pred3[selected], minlength=10)
                                    )

    total = len(numbers_all)
    result_bases = {}
    for basis_name, pca in bases.items():
        max_k = pca["max_k"]
        basis = pca["basis"][:, :max_k]
        result_bases[basis_name] = {
            "rank": pca["rank"],
            "tested_dimensions": max_k,
            "explained_variance_by_pc": [
                float(value) for value in pca["explained"][:max_k]
            ],
            "cumulative_explained_variance": [
                float(value) for value in pca["explained"][:max_k].cumsum(0)
            ],
            "readouts": {
                state: {
                    candidate_set: {
                        readout: serialize_curve(
                            correct[basis_name][state][candidate_set][readout], total
                        )
                        for readout in readout_types
                    }
                    for candidate_set in candidate_sets
                }
                for state in states
            },
            "digit_confusion_at_k3": {
                state: {
                    readout: by_max_k3[basis_name][state][readout].tolist()
                    for readout in readout_types
                }
                for state in states
            },
            "digit_projected_norms_at_k3": {
                "raw": [
                    float(value)
                    for value in (unembedding_cpu[:10] @ basis[:, :3]).norm(dim=1)
                ],
                "centered": [
                    float(value)
                    for value in (
                        (unembedding_cpu[:10] - unembedding_cpu[:10].mean(dim=0))
                        @ basis[:, :3]
                    ).norm(dim=1)
                ],
            },
            "max_abs_per_head_route_error": max_coordinate_error[basis_name],
            "max_row_error_in_raw_minus_centered_candidate_offset": (
                centered_raw_candidate_offset_error[basis_name]
            ),
        }
        if centered_raw_candidate_offset_error[basis_name] > 1e-5:
            raise AssertionError(
                f"centering did not produce a common candidate offset for "
                f"{basis_name}: {centered_raw_candidate_offset_error[basis_name]}"
            )

    q_u = bases["digit_unembedding"]["basis"][:, :3]
    q_o = bases["output_matrix"]["basis"][:, :3]
    overlap = q_u.T @ q_o
    principal_cosines = torch.linalg.svdvals(overlap).clamp(-1.0, 1.0)
    principal_angles = torch.rad2deg(torch.arccos(principal_cosines))
    return {
        "bases": result_bases,
        "digit_WU_vs_output_WO_top3": {
            "pc_cosine_matrix": [
                [float(value) for value in row] for row in overlap
            ],
            "principal_angles_degrees": [
                float(value) for value in principal_angles
            ],
        },
    }


def plot_summary(result: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)
    xs = np.arange(len(MODEL_ORDER))
    norm_means = [result["models"][name]["core"]["unembedding_row_norms"]["mean"] for name in MODEL_ORDER]
    norm_mins = [result["models"][name]["core"]["unembedding_row_norms"]["minimum"] for name in MODEL_ORDER]
    norm_maxes = [result["models"][name]["core"]["unembedding_row_norms"]["maximum"] for name in MODEL_ORDER]
    axes[0].errorbar(
        xs,
        norm_means,
        yerr=[np.asarray(norm_means) - np.asarray(norm_mins), np.asarray(norm_maxes) - np.asarray(norm_means)],
        fmt="o",
        color="#2563eb",
        capsize=5,
    )
    axes[0].axhline(1.0, color="#111827", linestyle="--", linewidth=1)
    axes[0].set_xticks(xs, [MODEL_LABELS[name] for name in MODEL_ORDER], rotation=18)
    axes[0].set_ylabel("Unembedding row norm")
    axes[0].set_title("All 14 W_U rows")
    axes[0].grid(axis="y", alpha=0.2)

    width = 0.36
    actual = [result["models"][name]["core"]["readout"]["actual"]["accuracy"] for name in MODEL_ORDER]
    head_sum = [result["models"][name]["core"]["readout"]["head_sum_only"]["accuracy"] for name in MODEL_ORDER]
    axes[1].bar(xs - width / 2, actual, width, label="residual + heads", color="#16a34a")
    axes[1].bar(xs + width / 2, head_sum, width, label="heads only", color="#7c3aed")
    axes[1].set_xticks(xs, [MODEL_LABELS[name] for name in MODEL_ORDER], rotation=18)
    axes[1].set_ylim(0, 1.06)
    axes[1].set_ylabel("Exhaustive accuracy")
    axes[1].set_title("100,000 inputs")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    dot = [result["models"][name]["core"]["readout"]["actual"]["accuracy"] for name in MODEL_ORDER]
    cosine = [result["models"][name]["core"]["readout"]["full64_cosine"]["accuracy"] for name in MODEL_ORDER]
    axes[2].bar(xs - width / 2, dot, width, label="dot product", color="#0f766e")
    axes[2].bar(xs + width / 2, cosine, width, label="cosine", color="#dc2626")
    axes[2].set_xticks(xs, [MODEL_LABELS[name] for name in MODEL_ORDER], rotation=18)
    axes[2].set_ylim(0, 1.06)
    axes[2].set_ylabel("Full-64D accuracy")
    axes[2].set_title("Actual final state")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("Puzzle 1a: unit-sphere unembedding retrains")
    fig.savefig(SUMMARY_PNG, dpi=180, facecolor="white")
    plt.close(fig)


def plot_lowdim(result: dict) -> None:
    basis_names = ("digit_unembedding", "full_unembedding", "output_matrix")
    titles = ("digit W_U PCA", "full-vocab W_U PCA", "stacked W_O PCA")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    for axis, basis_name, title in zip(axes, basis_names, titles):
        for model_name, linestyle in (("released_original", "--"), (PRIMARY_MODEL, "-")):
            basis = result["models"][model_name]["lowdim"]["bases"][basis_name]
            curve = basis["readouts"]["actual"]["digits"]["dot_raw"]["accuracy_by_k"]
            xs = np.arange(1, len(curve) + 1)
            axis.plot(
                xs,
                100 * np.asarray(curve),
                linestyle=linestyle,
                linewidth=2.2,
                color="#7c3aed" if model_name == PRIMARY_MODEL else "#64748b",
                label=MODEL_LABELS[model_name],
            )
        axis.axhline(100, color="#111827", linewidth=0.8, alpha=0.6)
        axis.axvline(3, color="#dc2626", linewidth=0.9, linestyle=":")
        axis.set_xlabel("PCs retained")
        axis.set_ylabel("Dot-product accuracy (%)")
        axis.set_ylim(0, 104)
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        if basis_name == "output_matrix":
            axis.set_xlim(0.8, 10.2)
    fig.suptitle("Actual residual + head state projected into PCA bases")
    fig.savefig(LOWDIM_PNG, dpi=180, facecolor="white")
    plt.close(fig)


def plot_recruitment(result: dict) -> None:
    core = result["models"][PRIMARY_MODEL]["core"]
    max_mass = np.zeros((4, 10))
    ans_mass = np.zeros((4, 10))
    force_ans_accuracy = np.zeros((4, 10))
    for maximum in range(10):
        for head in range(4):
            row = core["attention_by_true_max"][str(maximum)][f"H{head}"]
            max_mass[head, maximum] = row["max_mass"]["mean"]
            ans_mass[head, maximum] = row["ans_mass"]["mean"]
            force_ans_accuracy[head, maximum] = core["interventions"]["force_ANS"][f"H{head}"]["by_true_max"][str(maximum)]["accuracy"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    for axis, matrix, title, cmap, vmin, vmax in (
        (axes[0], max_mass, "Mean attention mass to max-valued positions", "Blues", 0, 1),
        (axes[1], ans_mass, "Mean attention mass to [ANS]", "Greens", 0, 1),
        (axes[2], force_ans_accuracy, "Accuracy after forcing one head to [ANS]", "RdYlGn", 0, 1),
    ):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_xticks(range(10))
        axis.set_yticks(range(4), [f"H{head}" for head in range(4)])
        axis.set_xlabel("True maximum")
        axis.set_title(title)
        for head in range(4):
            for maximum in range(10):
                value = matrix[head, maximum]
                axis.text(maximum, head, f"{value:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    fig.suptitle("Unit-unembedding seed 42: ANS-query head recruitment")
    fig.savefig(RECRUITMENT_PNG, dpi=180, facecolor="white")
    plt.close(fig)


def confusion_accuracy(confusion: list[list[int]]) -> np.ndarray:
    matrix = np.asarray(confusion)
    totals = matrix.sum(axis=1)
    return np.diag(matrix) / totals


def plot_angular(result: dict) -> None:
    basis_names = ("digit_unembedding", "full_unembedding", "output_matrix")
    titles = ("digit W_U PCs", "full-vocab W_U PCs", "stacked W_O PCs")
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
    digits = np.arange(10)
    for col, (basis_name, title) in enumerate(zip(basis_names, titles)):
        basis = result["models"][PRIMARY_MODEL]["lowdim"]["bases"][basis_name]
        confusions = basis["digit_confusion_at_k3"]["actual"]
        dot_accuracy = confusion_accuracy(confusions["dot_raw"])
        raw_cosine = confusion_accuracy(confusions["cosine_raw"])
        centered_cosine = confusion_accuracy(confusions["cosine_centered"])
        width = 0.25
        axes[0, col].bar(digits - width, dot_accuracy, width, label="dot", color="#0f766e")
        axes[0, col].bar(digits, raw_cosine, width, label="cos raw U", color="#2563eb")
        axes[0, col].bar(digits + width, centered_cosine, width, label="cos centered U", color="#dc2626")
        axes[0, col].set_xticks(digits)
        axes[0, col].set_ylim(0, 1.08)
        axes[0, col].set_xlabel("True maximum")
        axes[0, col].set_ylabel("Top-3 digit-PCA accuracy")
        axes[0, col].set_title(title)
        axes[0, col].legend(frameon=False, fontsize=8)
        axes[0, col].grid(axis="y", alpha=0.2)

        norms = basis["digit_projected_norms_at_k3"]
        axes[1, col].bar(digits - 0.18, norms["raw"], 0.36, label="raw projection", color="#2563eb")
        axes[1, col].bar(digits + 0.18, norms["centered"], 0.36, label="centered coordinate", color="#f97316")
        axes[1, col].set_xticks(digits)
        axes[1, col].set_xlabel("Digit")
        axes[1, col].set_ylabel("Projected unembedding norm")
        axes[1, col].set_title("Unequal norms can reappear after projection")
        axes[1, col].legend(frameon=False, fontsize=8)
        axes[1, col].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Unit-unembedding seed 42: dot product versus angular decoding in three-PC bases"
    )
    fig.savefig(ANGULAR_PNG, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numbers, labels = all_inputs()
    result = {
        "description": (
            "Released Puzzle 1a model versus three retrains whose 14 unembedding "
            "rows are projected to unit norm after initialization and every optimizer step."
        ),
        "scope": "all 100000 possible five-digit inputs",
        "primary_model": PRIMARY_MODEL,
        "counts_by_true_max": {
            str(maximum): int(EXPECTED_COUNTS[maximum]) for maximum in range(10)
        },
        "models": {},
    }
    for model_name in MODEL_ORDER:
        print(f"analyzing,{model_name}", flush=True)
        model, config = load_model(model_name, device)
        core = analyze_core(model, numbers, labels)
        lowdim = analyze_lowdim(model, numbers, labels)
        result["models"][model_name] = {
            "label": MODEL_LABELS[model_name],
            "config": config,
            "core": core,
            "lowdim": lowdim,
        }
        if model_name.startswith("unit_"):
            if core["unembedding_row_norms"]["max_abs_error_from_one"] > 1e-6:
                raise AssertionError(f"{model_name} does not have unit W_U rows")
            if core["readout"]["actual"]["correct"] != 100000:
                raise AssertionError(f"{model_name} is not exhaustively perfect")
            if core["readout"]["actual"]["correct"] != core["readout"]["full64_cosine"]["correct"]:
                raise AssertionError(f"{model_name} dot and cosine readouts disagree")

    JSON_OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot_summary(result)
    plot_lowdim(result)
    plot_recruitment(result)
    plot_angular(result)

    print("model,actual,head_sum,full64_cosine,WU_norm_min,WU_norm_max")
    for model_name in MODEL_ORDER:
        core = result["models"][model_name]["core"]
        print(
            f"{model_name},{core['readout']['actual']['accuracy']:.6f},"
            f"{core['readout']['head_sum_only']['accuracy']:.6f},"
            f"{core['readout']['full64_cosine']['accuracy']:.6f},"
            f"{core['unembedding_row_norms']['minimum']:.9f},"
            f"{core['unembedding_row_norms']['maximum']:.9f}"
        )
    print(f"wrote,{JSON_OUT}")
    for path in (SUMMARY_PNG, LOWDIM_PNG, RECRUITMENT_PNG, ANGULAR_PNG):
        print(f"wrote,{path}")


if __name__ == "__main__":
    main()
