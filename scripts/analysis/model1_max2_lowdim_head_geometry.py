#!/usr/bin/env python3
"""Plot per-head 3d writes and digit unembeddings for a true-max-2 input."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_max2_lowdim_head_geometry.png"
ARROW_OUT = ROOT / "docs" / "assets" / "model1_max2_lowdim_arrow_alignment.png"
COUNTERFACTUAL_OUT = ROOT / "docs" / "assets" / "model1_max2_h3_self_counterfactual_arrows.png"
LOW012_OUT = ROOT / "docs" / "assets" / "model1_low012_actual_head_arrows.png"
LOW0TO6_OUT = ROOT / "docs" / "assets" / "model1_low0to6_piecewise_head_arrows.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_max2_lowdim_head_geometry.json"
NUMBERS = [0, 1, 2, 0, 1]
NUMBER_POSITIONS = [1, 3, 5, 7, 9]
HEAD_LABELS = ["H0", "H1", "H2", "H3"]
HEAD_COLORS = ["#2563eb", "#f59e0b", "#16a34a", "#dc2626"]
UNEMBED_COLORS = [
    "#7c3aed",
    "#a16207",
    "#db2777",
    "#0891b2",
    "#65a30d",
    "#ea580c",
    "#4f46e5",
    "#0f766e",
    "#be123c",
    "#52525b",
]
SPECIAL_TOKENS = {10: "BOS", 11: "SEP", 12: "ANS", 13: "EOS"}


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


def tokenize(numbers: list[int]) -> torch.Tensor:
    return torch.tensor(
        [[10, numbers[0], 11, numbers[1], 11, numbers[2], 11, numbers[3], 11, numbers[4], 12]],
        dtype=torch.long,
    )


def source_label(token_id: int, position: int) -> str:
    token = str(token_id) if token_id < 10 else SPECIAL_TOKENS[token_id]
    return f"{token}@{position}"


def padded_limits(points: torch.Tensor) -> list[tuple[float, float]]:
    limits = []
    for dim in range(3):
        values = points[:, dim]
        low = min(float(values.min()), 0.0)
        high = max(float(values.max()), 0.0)
        span = max(high - low, 0.25)
        limits.append((low - 0.10 * span, high + 0.10 * span))
    return limits


def vector_relation(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    angle_degrees = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return {"cosine": cosine, "angle_degrees": angle_degrees}


def main() -> None:
    torch.manual_seed(0)
    model, config = load_model()
    tokens = tokenize(NUMBERS)
    positions = torch.arange(tokens.shape[1]).unsqueeze(0)
    resid = model.tok_embed(tokens) + model.pos_embed(positions)
    layer = model.layers[0]
    causal_mask = torch.tril(torch.ones(tokens.shape[1], tokens.shape[1])).unsqueeze(0)

    digit_unembedding = model.unembed.weight.detach()[:10]
    digit_mean = digit_unembedding.mean(dim=0)
    centered_digits = digit_unembedding - digit_mean
    _, singular_values, directions = torch.linalg.svd(centered_digits, full_matrices=False)
    basis = directions[:3]
    explained = singular_values.square() / singular_values.square().sum()
    digit_coordinates = centered_digits @ basis.T

    head_values = []
    head_coordinates = []
    output_matrices = []
    reduced_outputs = []
    attention_rows = []
    attention_summary = []
    h3_self_value = None
    w_o = layer.W_O.weight.detach()
    max_positions = [
        position for position, number in zip(NUMBER_POSITIONS, NUMBERS) if number == max(NUMBERS)
    ]

    with torch.no_grad():
        for head_idx, head in enumerate(layer.heads):
            output, attention = head(resid, causal_mask)
            value = output[0, 10]
            source_values = resid @ head.W_V.weight.detach().T
            if head_idx == 3:
                h3_self_value = source_values[0, 10]
            output_matrix = w_o[:, head_idx * head.d_head : (head_idx + 1) * head.d_head].T
            reduced_output = output_matrix @ basis.T
            coordinates = value @ reduced_output
            attention_row = attention[0, 10]
            top_position = int(attention_row.argmax())

            head_values.append(value)
            output_matrices.append(output_matrix)
            reduced_outputs.append(reduced_output)
            head_coordinates.append(coordinates)
            attention_rows.append(attention_row)
            attention_summary.append(
                {
                    "head": f"H{head_idx}",
                    "top_position": top_position,
                    "top_source": source_label(int(tokens[0, top_position]), top_position),
                    "top_attention": float(attention_row[top_position]),
                    "ans_self_attention": float(attention_row[10]),
                    "max_digit_attention": float(attention_row[max_positions].sum()),
                }
            )

    head_values_t = torch.stack(head_values)
    output_matrices_t = torch.stack(output_matrices)
    reduced_outputs_t = torch.stack(reduced_outputs)
    head_coordinates_t = torch.stack(head_coordinates)
    attention_rows_t = torch.stack(attention_rows)
    summed_coordinates = head_coordinates_t.sum(dim=0)
    low_dim_logits = summed_coordinates @ digit_coordinates.T
    prediction = int(low_dim_logits.argmax())
    runner_up = int(torch.topk(low_dim_logits, 2).indices[1])
    margin = float(low_dim_logits[prediction] - low_dim_logits[runner_up])
    full_head_writes = torch.einsum("hd,hdm->hm", head_values_t, output_matrices_t)
    full_summed_write = full_head_writes.sum(dim=0)
    full_digit_logits = full_summed_write @ digit_unembedding.T
    full_prediction = int(full_digit_logits.argmax())

    if h3_self_value is None:
        raise AssertionError("H3 self value was not collected")
    counterfactual_head_values = head_values_t.clone()
    counterfactual_head_values[3] = h3_self_value
    counterfactual_head_coordinates = torch.einsum(
        "hd,hdk->hk", counterfactual_head_values, reduced_outputs_t
    )
    counterfactual_sum = counterfactual_head_coordinates.sum(dim=0)
    counterfactual_logits = counterfactual_sum @ digit_coordinates.T
    counterfactual_prediction = int(counterfactual_logits.argmax())
    counterfactual_runner_up = int(torch.topk(counterfactual_logits, 2).indices[1])
    counterfactual_margin = float(
        counterfactual_logits[counterfactual_prediction]
        - counterfactual_logits[counterfactual_runner_up]
    )
    counterfactual_full_head_writes = torch.einsum(
        "hd,hdm->hm", counterfactual_head_values, output_matrices_t
    )
    counterfactual_full_summed_write = counterfactual_full_head_writes.sum(dim=0)
    counterfactual_full_digit_logits = counterfactual_full_summed_write @ digit_unembedding.T
    counterfactual_full_prediction = int(counterfactual_full_digit_logits.argmax())

    expected_top_positions = [10, 10, 10, 5]
    actual_top_positions = [row["top_position"] for row in attention_summary]
    if actual_top_positions != expected_top_positions:
        raise AssertionError(f"unexpected attention destinations: {actual_top_positions}")
    if prediction != 2:
        raise AssertionError(f"three-dimensional readout predicted {prediction}, expected 2")
    if full_prediction != 2:
        raise AssertionError(f"full 64d head-sum readout predicted {full_prediction}, expected 2")
    if counterfactual_prediction != 0:
        raise AssertionError(
            f"H3-to-ANS counterfactual predicted {counterfactual_prediction}, expected 0"
        )
    if counterfactual_full_prediction != 0:
        raise AssertionError(
            "full 64d H3-to-ANS counterfactual predicted "
            f"{counterfactual_full_prediction}, expected 0"
        )
    if head_values_t.shape != (4, 16) or reduced_outputs_t.shape != (4, 16, 3):
        raise AssertionError("unexpected value or reduced-output shape")

    target_cosine = float(
        torch.dot(summed_coordinates, digit_coordinates[2])
        / (summed_coordinates.norm() * digit_coordinates[2].norm())
    )
    cosine_by_digit = torch.nn.functional.cosine_similarity(
        summed_coordinates.unsqueeze(0), digit_coordinates, dim=1
    )
    nearest_digit = int(cosine_by_digit.argmax())
    counterfactual_cosine_by_digit = torch.nn.functional.cosine_similarity(
        counterfactual_sum.unsqueeze(0), digit_coordinates, dim=1
    )
    counterfactual_nearest_digit = int(counterfactual_cosine_by_digit.argmax())

    max_digit_radius = float(digit_coordinates.norm(dim=1).max())
    all_head_coordinates = torch.cat([head_coordinates_t, summed_coordinates.unsqueeze(0)])
    max_head_radius = float(all_head_coordinates.norm(dim=1).max())
    head_display_scale = 0.90 * max_digit_radius / max_head_radius
    displayed_heads = head_coordinates_t * head_display_scale
    displayed_sum = summed_coordinates * head_display_scale

    comparison_vectors = torch.cat(
        [
            head_coordinates_t,
            summed_coordinates.unsqueeze(0),
            counterfactual_head_coordinates,
            counterfactual_sum.unsqueeze(0),
        ]
    )
    comparison_display_scale = (
        0.90 * max_digit_radius / float(comparison_vectors.norm(dim=1).max())
    )

    matched_case_head_coordinates = []
    matched_case_sums = []
    matched_case_full_head_writes = []
    matched_low_cases = []
    with torch.no_grad():
        for true_max in range(3):
            case_numbers = [0, 0, true_max, 0, 0]
            case_tokens = tokenize(case_numbers)
            case_positions = torch.arange(case_tokens.shape[1]).unsqueeze(0)
            case_resid = model.tok_embed(case_tokens) + model.pos_embed(case_positions)
            case_mask = torch.tril(
                torch.ones(case_tokens.shape[1], case_tokens.shape[1])
            ).unsqueeze(0)
            actual_logits, _ = model(case_tokens)

            case_values = []
            case_attention_summary = []
            max_number_positions = [
                position
                for position, number in zip(NUMBER_POSITIONS, case_numbers)
                if number == true_max
            ]
            for head_idx, head in enumerate(layer.heads):
                case_output, case_attention = head(case_resid, case_mask)
                case_values.append(case_output[0, 10])
                case_row = case_attention[0, 10]
                top_position = int(case_row.argmax())
                case_attention_summary.append(
                    {
                        "head": f"H{head_idx}",
                        "top_position": top_position,
                        "top_source": source_label(
                            int(case_tokens[0, top_position]), top_position
                        ),
                        "top_attention": float(case_row[top_position]),
                        "ans_self_attention": float(case_row[10]),
                        "max_digit_attention": float(case_row[max_number_positions].sum()),
                        "attention_row": [float(value) for value in case_row],
                    }
                )

            case_values_t = torch.stack(case_values)
            case_head_coordinates = torch.einsum(
                "hd,hdk->hk", case_values_t, reduced_outputs_t
            )
            case_sum = case_head_coordinates.sum(dim=0)
            case_low_dim_logits = case_sum @ digit_coordinates.T
            case_prediction = int(case_low_dim_logits.argmax())
            case_top2 = torch.topk(case_low_dim_logits, 2)
            case_runner_up = int(case_top2.indices[1])
            case_margin = float(case_top2.values[0] - case_top2.values[1])

            case_full_head_writes = torch.einsum(
                "hd,hdm->hm", case_values_t, output_matrices_t
            )
            case_full_logits = case_full_head_writes.sum(dim=0) @ digit_unembedding.T
            case_full_prediction = int(case_full_logits.argmax())
            case_actual_prediction = int(actual_logits[0, 10, :10].argmax())
            if not (
                case_prediction
                == case_full_prediction
                == case_actual_prediction
                == true_max
            ):
                raise AssertionError(
                    "matched low-digit case predictions disagree: "
                    f"max={true_max}, 3d={case_prediction}, 64d={case_full_prediction}, "
                    f"model={case_actual_prediction}"
                )

            angle_comparisons = {}
            for head_idx in range(3):
                angle_comparisons[f"H{head_idx}_vs_H3"] = {
                    "projected_3d": vector_relation(
                        case_head_coordinates[head_idx], case_head_coordinates[3]
                    ),
                    "full_64d": vector_relation(
                        case_full_head_writes[head_idx], case_full_head_writes[3]
                    ),
                }
            angle_comparisons["H0_H1_H2_sum_vs_H3"] = {
                "projected_3d": vector_relation(
                    case_head_coordinates[:3].sum(dim=0), case_head_coordinates[3]
                ),
                "full_64d": vector_relation(
                    case_full_head_writes[:3].sum(dim=0), case_full_head_writes[3]
                ),
            }

            h3_summary = case_attention_summary[3]
            matched_case_head_coordinates.append(case_head_coordinates)
            matched_case_sums.append(case_sum)
            matched_case_full_head_writes.append(case_full_head_writes)
            matched_low_cases.append(
                {
                    "true_max": true_max,
                    "numbers": case_numbers,
                    "tokens": case_tokens.squeeze(0).tolist(),
                    "attention_summary": case_attention_summary,
                    "h3_ans_self_attention": h3_summary["ans_self_attention"],
                    "h3_max_digit_attention": h3_summary["max_digit_attention"],
                    "h3_other_attention": (
                        1.0
                        - h3_summary["ans_self_attention"]
                        - h3_summary["max_digit_attention"]
                    ),
                    "head_pc_coordinates": {
                        f"H{head_idx}": [
                            float(value) for value in case_head_coordinates[head_idx]
                        ]
                        for head_idx in range(4)
                    },
                    "summed_pc_coordinates": [float(value) for value in case_sum],
                    "low_dimensional_relative_logits": [
                        float(value) for value in case_low_dim_logits
                    ],
                    "prediction": case_prediction,
                    "runner_up": case_runner_up,
                    "prediction_margin": case_margin,
                    "full_64d_head_sum_prediction": case_full_prediction,
                    "actual_model_prediction": case_actual_prediction,
                    "angle_comparisons": angle_comparisons,
                }
            )

    matched_case_head_coordinates_t = torch.stack(matched_case_head_coordinates)
    matched_case_sums_t = torch.stack(matched_case_sums)
    matched_case_full_head_writes_t = torch.stack(matched_case_full_head_writes)
    matched_case_vectors = torch.cat(
        [matched_case_head_coordinates_t.flatten(0, 1), matched_case_sums_t]
    )
    matched_case_display_scale = (
        0.90 * max_digit_radius / float(matched_case_vectors.norm(dim=1).max())
    )
    h3_direction_stability = {}
    for first_max, second_max in [(0, 1), (0, 2), (1, 2)]:
        h3_direction_stability[f"max_{first_max}_vs_max_{second_max}"] = {
            "projected_3d": vector_relation(
                matched_case_head_coordinates_t[first_max, 3],
                matched_case_head_coordinates_t[second_max, 3],
            ),
            "full_64d": vector_relation(
                matched_case_full_head_writes_t[first_max, 3],
                matched_case_full_head_writes_t[second_max, 3],
            ),
        }

    piecewise_case_head_coordinates = []
    piecewise_case_sums = []
    piecewise_case_full_head_writes = []
    piecewise_cases = []
    with torch.no_grad():
        for true_max in range(7):
            case_numbers = [0, 0, true_max, 0, 0]
            case_tokens = tokenize(case_numbers)
            case_positions = torch.arange(case_tokens.shape[1]).unsqueeze(0)
            case_resid = model.tok_embed(case_tokens) + model.pos_embed(case_positions)
            case_mask = torch.tril(
                torch.ones(case_tokens.shape[1], case_tokens.shape[1])
            ).unsqueeze(0)
            actual_logits, _ = model(case_tokens)

            self_values = []
            source_values = []
            actual_values = []
            h3_attention_row = None
            for head_idx, head in enumerate(layer.heads):
                case_output, case_attention = head(case_resid, case_mask)
                head_source_values = case_resid @ head.W_V.weight.detach().T
                source_values.append(head_source_values[0])
                self_values.append(head_source_values[0, 10])
                actual_values.append(case_output[0, 10])
                if head_idx == 3:
                    h3_attention_row = case_attention[0, 10]

            if true_max == 0:
                chosen_values = self_values
                recipe = {
                    "H0": "ANS self one-hot",
                    "H1": "ANS self one-hot",
                    "H2": "ANS self one-hot",
                    "H3": "ANS self one-hot",
                }
            elif true_max == 1:
                chosen_values = self_values[:3] + [actual_values[3]]
                recipe = {
                    "H0": "ANS self one-hot",
                    "H1": "ANS self one-hot",
                    "H2": "ANS self one-hot",
                    "H3": "actual soft ANS + max-1 mixture",
                }
            else:
                chosen_values = self_values[:3] + [source_values[3][5]]
                recipe = {
                    "H0": "ANS self one-hot",
                    "H1": "ANS self one-hot",
                    "H2": "ANS self one-hot",
                    "H3": f"{true_max}@5 one-hot",
                }

            chosen_values_t = torch.stack(chosen_values)
            case_head_coordinates = torch.einsum(
                "hd,hdk->hk", chosen_values_t, reduced_outputs_t
            )
            case_sum = case_head_coordinates.sum(dim=0)
            case_logits = case_sum @ digit_coordinates.T
            case_top2 = torch.topk(case_logits, 2)
            case_prediction = int(case_top2.indices[0])
            case_runner_up = int(case_top2.indices[1])
            case_margin = float(case_top2.values[0] - case_top2.values[1])

            case_full_head_writes = torch.einsum(
                "hd,hdm->hm", chosen_values_t, output_matrices_t
            )
            case_full_logits = case_full_head_writes.sum(dim=0) @ digit_unembedding.T
            case_full_prediction = int(case_full_logits.argmax())
            case_actual_prediction = int(actual_logits[0, 10, :10].argmax())
            if not (
                case_prediction
                == case_full_prediction
                == case_actual_prediction
                == true_max
            ):
                raise AssertionError(
                    "piecewise low-max predictions disagree: "
                    f"max={true_max}, 3d={case_prediction}, 64d={case_full_prediction}, "
                    f"model={case_actual_prediction}"
                )
            if h3_attention_row is None:
                raise AssertionError("H3 attention row was not collected")

            selected_unembeddings = (
                [0, 1, 2, 3]
                if true_max <= 2
                else [true_max - 2, true_max - 1, true_max, true_max + 1]
            )
            piecewise_case_head_coordinates.append(case_head_coordinates)
            piecewise_case_sums.append(case_sum)
            piecewise_case_full_head_writes.append(case_full_head_writes)
            piecewise_cases.append(
                {
                    "true_max": true_max,
                    "numbers": case_numbers,
                    "tokens": case_tokens.squeeze(0).tolist(),
                    "recipe": recipe,
                    "selected_unembedding_digits": selected_unembeddings,
                    "h3_actual_attention_row": [
                        float(value) for value in h3_attention_row
                    ],
                    "head_pc_coordinates": {
                        f"H{head_idx}": [
                            float(value) for value in case_head_coordinates[head_idx]
                        ]
                        for head_idx in range(4)
                    },
                    "summed_pc_coordinates": [float(value) for value in case_sum],
                    "low_dimensional_relative_logits": [
                        float(value) for value in case_logits
                    ],
                    "prediction": case_prediction,
                    "runner_up": case_runner_up,
                    "prediction_margin": case_margin,
                    "full_64d_head_sum_prediction": case_full_prediction,
                    "actual_model_prediction": case_actual_prediction,
                }
            )

    piecewise_case_head_coordinates_t = torch.stack(piecewise_case_head_coordinates)
    piecewise_case_sums_t = torch.stack(piecewise_case_sums)
    piecewise_case_full_head_writes_t = torch.stack(piecewise_case_full_head_writes)
    piecewise_vectors = torch.cat(
        [piecewise_case_head_coordinates_t.flatten(0, 1), piecewise_case_sums_t]
    )
    piecewise_display_scale = (
        0.90 * max_digit_radius / float(piecewise_vectors.norm(dim=1).max())
    )
    piecewise_h3_axis_relations = {}
    for true_max in range(7):
        piecewise_h3_axis_relations[str(true_max)] = {
            "projected_3d_vs_max_0": vector_relation(
                piecewise_case_head_coordinates_t[0, 3],
                piecewise_case_head_coordinates_t[true_max, 3],
            ),
            "full_64d_vs_max_0": vector_relation(
                piecewise_case_full_head_writes_t[0, 3],
                piecewise_case_full_head_writes_t[true_max, 3],
            ),
            "projected_3d_norm": float(
                piecewise_case_head_coordinates_t[true_max, 3].norm()
            ),
            "full_64d_norm": float(
                piecewise_case_full_head_writes_t[true_max, 3].norm()
            ),
        }

    result = {
        "description": (
            "Actual Model 1 post-attention pre-W_O values for input [0,1,2,0,1]. "
            "Each value_h (1x16) is mapped by O_h @ P3 (16x3), where P3 is the "
            "digit-only unembedding PCA basis. The four 3d writes are summed and "
            "multiplied by the centered 3d digit unembedding."
        ),
        "hf_repo": "andyrdt/04_2026_puzzle_1a",
        "model_config": config,
        "numbers": NUMBERS,
        "tokens": tokens.squeeze(0).tolist(),
        "true_max": 2,
        "digit_pca_explained_variance": [float(value) for value in explained[:3]],
        "digit_pc_coordinates": [[float(value) for value in row] for row in digit_coordinates],
        "head_values_shape": list(head_values_t.shape),
        "reduced_output_shape": list(reduced_outputs_t.shape),
        "head_values": [[float(value) for value in row] for row in head_values_t],
        "reduced_output_matrices": {
            f"H{head_idx}": [
                [float(value) for value in row] for row in reduced_outputs_t[head_idx]
            ]
            for head_idx in range(4)
        },
        "head_pc_coordinates": {
            f"H{head_idx}": [float(value) for value in head_coordinates_t[head_idx]]
            for head_idx in range(4)
        },
        "summed_pc_coordinates": [float(value) for value in summed_coordinates],
        "head_display_scale": head_display_scale,
        "attention_rows": {
            f"H{head_idx}": [float(value) for value in attention_rows_t[head_idx]]
            for head_idx in range(4)
        },
        "attention_summary": attention_summary,
        "low_dimensional_relative_logits": [float(value) for value in low_dim_logits],
        "prediction": prediction,
        "full_64d_head_sum_digit_logits": [float(value) for value in full_digit_logits],
        "full_64d_head_sum_prediction": full_prediction,
        "runner_up": runner_up,
        "prediction_margin": margin,
        "cosine_sum_vs_true_max": target_cosine,
        "nearest_digit_by_cosine": nearest_digit,
        "cosine_by_digit": [float(value) for value in cosine_by_digit],
        "h3_to_ans_counterfactual": {
            "intervention": "Replace only H3's ANS attention row by one-hot attention to ANS@10.",
            "h3_self_value": [float(value) for value in h3_self_value],
            "head_pc_coordinates": {
                f"H{head_idx}": [
                    float(value) for value in counterfactual_head_coordinates[head_idx]
                ]
                for head_idx in range(4)
            },
            "summed_pc_coordinates": [float(value) for value in counterfactual_sum],
            "low_dimensional_relative_logits": [
                float(value) for value in counterfactual_logits
            ],
            "prediction": counterfactual_prediction,
            "full_64d_head_sum_digit_logits": [
                float(value) for value in counterfactual_full_digit_logits
            ],
            "full_64d_head_sum_prediction": counterfactual_full_prediction,
            "runner_up": counterfactual_runner_up,
            "prediction_margin": counterfactual_margin,
            "nearest_digit_by_cosine": counterfactual_nearest_digit,
            "cosine_by_digit": [float(value) for value in counterfactual_cosine_by_digit],
            "comparison_display_scale": comparison_display_scale,
        },
        "matched_actual_low_max_cases": {
            "description": (
                "Three actual-model cases differing only in the center number: "
                "[0,0,0,0,0], [0,0,1,0,0], and [0,0,2,0,0]."
            ),
            "display_scale": matched_case_display_scale,
            "angle_preservation_scope": (
                "Descriptive check on these three matched examples only; PCA preserves "
                "variance, not pairwise angles."
            ),
            "digit_unembedding_top3_explained_variance": float(explained[:3].sum()),
            "h3_direction_stability": h3_direction_stability,
            "cases": matched_low_cases,
        },
        "piecewise_low_max_0_to_6": {
            "description": (
                "Verified attention abstraction on matched inputs [0,0,max,0,0]: "
                "all heads self-read for max 0; H0-H2 self-read and H3 retains its "
                "actual soft mix for max 1; H0-H2 self-read and H3 one-hot reads the "
                "unique maximum for max 2 through 6."
            ),
            "display_scale": piecewise_display_scale,
            "h3_axis_relations": piecewise_h3_axis_relations,
            "cases": piecewise_cases,
        },
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    fig = plt.figure(figsize=(15.0, 7.2), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    bar_ax = fig.add_subplot(1, 2, 2)

    digit_np = digit_coordinates.numpy()
    ax.plot(
        digit_np[:, 0],
        digit_np[:, 1],
        digit_np[:, 2],
        color="#6b7280",
        linewidth=1.0,
        alpha=0.65,
        label="digit order",
    )
    ax.scatter(
        digit_np[:, 0],
        digit_np[:, 1],
        digit_np[:, 2],
        c=np.arange(10),
        cmap="viridis",
        s=58,
        edgecolor="white",
        linewidth=0.7,
        depthshade=False,
        label="centered digit unembeddings",
    )
    ax.scatter(
        digit_np[2, 0],
        digit_np[2, 1],
        digit_np[2, 2],
        facecolors="none",
        edgecolors="#111827",
        marker="o",
        s=165,
        linewidth=1.6,
        depthshade=False,
        label="true max 2",
    )
    spans = np.maximum(np.ptp(digit_np, axis=0), 1e-6)
    for digit in range(10):
        ax.text(
            digit_np[digit, 0] + 0.018 * spans[0],
            digit_np[digit, 1],
            digit_np[digit, 2] + 0.018 * spans[2],
            str(digit),
            fontsize=8,
        )

    for head_idx, label in enumerate(HEAD_LABELS):
        coords = displayed_heads[head_idx].numpy()
        ax.plot(
            [0.0, coords[0]],
            [0.0, coords[1]],
            [0.0, coords[2]],
            color=HEAD_COLORS[head_idx],
            linewidth=0.9,
            alpha=0.28,
        )
        ax.scatter(
            coords[0],
            coords[1],
            coords[2],
            color=HEAD_COLORS[head_idx],
            marker="x",
            s=90,
            linewidth=2.3,
            depthshade=False,
            label=label,
        )
    sum_np = displayed_sum.numpy()
    ax.plot(
        [0.0, sum_np[0]],
        [0.0, sum_np[1]],
        [0.0, sum_np[2]],
        color="#111827",
        linewidth=1.1,
        alpha=0.4,
    )
    ax.scatter(
        sum_np[0],
        sum_np[1],
        sum_np[2],
        color="#111827",
        marker="*",
        s=175,
        depthshade=False,
        label="H0+H1+H2+H3",
    )
    ax.scatter(0.0, 0.0, 0.0, color="#6b7280", marker="o", s=18, depthshade=False)

    all_display_points = torch.cat(
        [digit_coordinates, displayed_heads, displayed_sum.unsqueeze(0), torch.zeros(1, 3)]
    )
    limits = padded_limits(all_display_points)
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_zlim(limits[2])
    ax.set_xlabel("PC1", labelpad=7)
    ax.set_ylabel("PC2", labelpad=7)
    ax.set_zlabel("PC3", labelpad=7)
    ax.set_title(
        f"Digit-only top-3 PCA | heads x {head_display_scale:.5f}\n"
        "input [0, 1, 2, 0, 1], true max 2",
        pad=14,
    )
    attention_text = " | ".join(
        f"{row['head']}->{row['top_source']} ({row['top_attention']:.3f})"
        for row in attention_summary
    )
    ax.text2D(0.01, 0.01, attention_text, transform=ax.transAxes, fontsize=7.5)
    ax.view_init(elev=22, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.15, 1.0, 0.85))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)

    colors = ["#6b7280"] * 10
    colors[2] = "#dc2626"
    bars = bar_ax.bar(range(10), low_dim_logits.numpy(), color=colors, alpha=0.88)
    bar_ax.axhline(0.0, color="#111827", linewidth=0.8)
    bar_ax.set_xticks(range(10))
    bar_ax.set_xlabel("Digit")
    bar_ax.set_ylabel("Centered relative logit from 3D head sum")
    bar_ax.set_title(f"Direct 3D readout predicts 2\nmargin over digit {runner_up}: {margin:.3f}")
    bar_ax.grid(axis="y", alpha=0.22)
    for digit, bar in enumerate(bars):
        value = float(low_dim_logits[digit])
        bar_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (1.0 if value >= 0 else -1.0),
            f"{value:.1f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7,
        )

    fig.suptitle("Model 1: max-2 head writes in the three-dimensional answer space", fontsize=15)
    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)

    arrow_fig = plt.figure(figsize=(11.5, 8.5), constrained_layout=True)
    arrow_ax = arrow_fig.add_subplot(1, 1, 1, projection="3d")

    arrow_vectors = []
    label_offsets = {
        "H0": (0.03, -0.02, 0.03),
        "H1": (-0.08, -0.04, 0.03),
        "H2": (0.03, 0.00, -0.04),
        "H3": (0.03, 0.04, 0.02),
        "SUM": (0.03, -0.03, 0.02),
        "U0": (0.03, 0.02, 0.03),
        "U1": (0.03, 0.02, 0.03),
        "U2": (0.03, 0.02, 0.03),
        "U3": (0.03, -0.03, 0.02),
    }

    def draw_arrow(vector: np.ndarray, color: str, label: str, linewidth: float) -> None:
        arrow_ax.quiver(
            0.0,
            0.0,
            0.0,
            vector[0],
            vector[1],
            vector[2],
            color=color,
            linewidth=linewidth,
            arrow_length_ratio=0.10,
            normalize=False,
        )
        offset = label_offsets[label]
        arrow_ax.text(
            vector[0] + offset[0],
            vector[1] + offset[1],
            vector[2] + offset[2],
            label,
            color=color,
            fontsize=10,
            weight="bold",
        )
        arrow_vectors.append(torch.tensor(vector))

    for head_idx, label in enumerate(HEAD_LABELS):
        draw_arrow(displayed_heads[head_idx].numpy(), HEAD_COLORS[head_idx], label, 2.2)
    draw_arrow(displayed_sum.numpy(), "#111827", "SUM", 3.2)
    for digit in range(4):
        width = 3.2 if digit == 2 else 2.1
        draw_arrow(digit_coordinates[digit].numpy(), UNEMBED_COLORS[digit], f"U{digit}", width)

    arrow_ax.scatter(0.0, 0.0, 0.0, color="#111827", marker="o", s=30, depthshade=False)
    arrow_limits = padded_limits(torch.stack(arrow_vectors + [torch.zeros(3)]))
    arrow_ax.set_xlim(arrow_limits[0])
    arrow_ax.set_ylim(arrow_limits[1])
    arrow_ax.set_zlim(arrow_limits[2])
    arrow_ax.set_xlabel("PC1", labelpad=8)
    arrow_ax.set_ylabel("PC2", labelpad=8)
    arrow_ax.set_zlabel("PC3", labelpad=8)
    arrow_ax.set_title(
        f"Heads use display scale x {head_display_scale:.5f}; centered U0-U3 are unscaled\n"
        "input [0, 1, 2, 0, 1], true max 2",
        pad=15,
    )
    arrow_ax.text2D(
        0.01,
        0.01,
        "dot(SUM,U0..U3) = [45.8, 89.8, 104.8, 90.2]",
        transform=arrow_ax.transAxes,
        fontsize=9,
    )
    arrow_ax.view_init(elev=22, azim=-58)
    arrow_ax.set_proj_type("ortho")
    arrow_ax.set_box_aspect((1.15, 1.0, 0.85))
    arrow_ax.grid(alpha=0.2)

    legend_handles = [
        Line2D([0], [0], color=HEAD_COLORS[idx], lw=2.2, label=label)
        for idx, label in enumerate(HEAD_LABELS)
    ]
    legend_handles.append(Line2D([0], [0], color="#111827", lw=3.2, label="SUM"))
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=UNEMBED_COLORS[digit],
            lw=3.2 if digit == 2 else 2.1,
            label=f"U{digit}",
        )
        for digit in range(4)
    )
    arrow_ax.legend(handles=legend_handles, loc="upper left", frameon=False, ncol=2, fontsize=8)
    arrow_fig.suptitle(
        "Model 1: colored head and unembedding arrows in the max-2 answer space",
        fontsize=15,
    )
    arrow_fig.savefig(ARROW_OUT, dpi=180, facecolor="white")
    plt.close(arrow_fig)

    comparison_fig = plt.figure(figsize=(16.0, 7.4), constrained_layout=True)
    comparison_axes = [
        comparison_fig.add_subplot(1, 2, panel + 1, projection="3d") for panel in range(2)
    ]
    actual_comparison_heads = head_coordinates_t * comparison_display_scale
    actual_comparison_sum = summed_coordinates * comparison_display_scale
    counterfactual_comparison_heads = (
        counterfactual_head_coordinates * comparison_display_scale
    )
    counterfactual_comparison_sum = counterfactual_sum * comparison_display_scale
    comparison_points = torch.cat(
        [
            digit_coordinates[:4],
            actual_comparison_heads,
            actual_comparison_sum.unsqueeze(0),
            counterfactual_comparison_heads,
            counterfactual_comparison_sum.unsqueeze(0),
            torch.zeros(1, 3),
        ]
    )
    comparison_limits = padded_limits(comparison_points)

    def draw_comparison_panel(
        panel_ax,
        displayed_head_coordinates: torch.Tensor,
        displayed_sum_coordinates: torch.Tensor,
        raw_sum_coordinates: torch.Tensor,
        panel_title: str,
        target_digit: int,
        axis_limits: list[tuple[float, float]],
        unembedding_digits: tuple[int, ...] = (0, 1, 2, 3),
    ) -> None:
        for head_idx, label in enumerate(HEAD_LABELS):
            vector = displayed_head_coordinates[head_idx].numpy()
            panel_ax.quiver(
                0.0,
                0.0,
                0.0,
                vector[0],
                vector[1],
                vector[2],
                color=HEAD_COLORS[head_idx],
                linewidth=2.2,
                arrow_length_ratio=0.10,
                normalize=False,
            )
            offset = label_offsets[label]
            panel_ax.text(
                vector[0] + offset[0],
                vector[1] + offset[1],
                vector[2] + offset[2],
                label,
                color=HEAD_COLORS[head_idx],
                fontsize=9,
                weight="bold",
            )

        sum_vector = displayed_sum_coordinates.numpy()
        panel_ax.quiver(
            0.0,
            0.0,
            0.0,
            sum_vector[0],
            sum_vector[1],
            sum_vector[2],
            color="#111827",
            linewidth=3.2,
            arrow_length_ratio=0.10,
            normalize=False,
        )
        sum_offset = label_offsets["SUM"]
        panel_ax.text(
            sum_vector[0] + sum_offset[0],
            sum_vector[1] + sum_offset[1],
            sum_vector[2] + sum_offset[2],
            "SUM",
            color="#111827",
            fontsize=9,
            weight="bold",
        )

        for digit in unembedding_digits:
            vector = digit_coordinates[digit].numpy()
            linewidth = 3.4 if digit == target_digit else 1.9
            alpha = 1.0 if digit == target_digit else 0.78
            panel_ax.quiver(
                0.0,
                0.0,
                0.0,
                vector[0],
                vector[1],
                vector[2],
                color=UNEMBED_COLORS[digit],
                linewidth=linewidth,
                alpha=alpha,
                arrow_length_ratio=0.10,
                normalize=False,
            )
            offset = label_offsets.get(f"U{digit}", (0.03, 0.02, 0.03))
            panel_ax.text(
                vector[0] + offset[0],
                vector[1] + offset[1],
                vector[2] + offset[2],
                f"U{digit}",
                color=UNEMBED_COLORS[digit],
                fontsize=9,
                weight="bold" if digit == target_digit else "normal",
            )

        panel_logits = raw_sum_coordinates @ digit_coordinates.T
        displayed_logits = ", ".join(
            f"U{digit}: {float(panel_logits[digit]):.1f}"
            for digit in unembedding_digits
        )
        panel_ax.text2D(
            0.01,
            0.01,
            f"selected dot products = [{displayed_logits}]",
            transform=panel_ax.transAxes,
            fontsize=8,
        )
        panel_ax.scatter(
            0.0, 0.0, 0.0, color="#111827", marker="o", s=24, depthshade=False
        )
        panel_ax.set_xlim(axis_limits[0])
        panel_ax.set_ylim(axis_limits[1])
        panel_ax.set_zlim(axis_limits[2])
        panel_ax.set_xlabel("PC1", labelpad=7)
        panel_ax.set_ylabel("PC2", labelpad=7)
        panel_ax.set_zlabel("PC3", labelpad=7)
        panel_ax.set_title(panel_title, pad=13)
        panel_ax.view_init(elev=22, azim=-58)
        panel_ax.set_proj_type("ortho")
        panel_ax.set_box_aspect((1.15, 1.0, 0.85))
        panel_ax.grid(alpha=0.2)

    draw_comparison_panel(
        comparison_axes[0],
        actual_comparison_heads,
        actual_comparison_sum,
        summed_coordinates,
        f"Actual: H3 reads 2@5\n3D prediction {prediction}, margin {margin:.2f}",
        2,
        comparison_limits,
    )
    draw_comparison_panel(
        comparison_axes[1],
        counterfactual_comparison_heads,
        counterfactual_comparison_sum,
        counterfactual_sum,
        (
            "Counterfactual: H3 [ANS] query -> [ANS] key (self)\n"
            f"3D prediction {counterfactual_prediction}, margin {counterfactual_margin:.2f}"
        ),
        0,
        comparison_limits,
    )
    comparison_axes[0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        ncol=2,
        fontsize=7.5,
    )
    comparison_fig.suptitle(
        "Model 1: changing only H3's ANS-row source moves the 3D answer write",
        fontsize=15,
    )
    comparison_fig.supxlabel(
        f"All head and SUM arrows use the same display scale x {comparison_display_scale:.5f}; "
        "centered U0-U3 are unscaled",
        fontsize=9,
    )
    comparison_fig.savefig(COUNTERFACTUAL_OUT, dpi=180, facecolor="white")
    plt.close(comparison_fig)

    low012_fig = plt.figure(figsize=(23.0, 7.3), constrained_layout=True)
    low012_axes = [
        low012_fig.add_subplot(1, 3, panel + 1, projection="3d") for panel in range(3)
    ]
    displayed_matched_heads = (
        matched_case_head_coordinates_t * matched_case_display_scale
    )
    displayed_matched_sums = matched_case_sums_t * matched_case_display_scale
    low012_points = torch.cat(
        [
            digit_coordinates[:4],
            displayed_matched_heads.flatten(0, 1),
            displayed_matched_sums,
            torch.zeros(1, 3),
        ]
    )
    low012_limits = padded_limits(low012_points)

    for true_max, case in enumerate(matched_low_cases):
        h3_ans_percent = 100.0 * case["h3_ans_self_attention"]
        h3_max_percent = 100.0 * case["h3_max_digit_attention"]
        max_source_label = "all 0 positions" if true_max == 0 else f"{true_max}@5"
        draw_comparison_panel(
            low012_axes[true_max],
            displayed_matched_heads[true_max],
            displayed_matched_sums[true_max],
            matched_case_sums_t[true_max],
            (
                f"Max {true_max}: {case['numbers']}\n"
                f"H3: {h3_ans_percent:.2f}% [ANS] + "
                f"{h3_max_percent:.2f}% {max_source_label}\n"
                f"3D prediction {case['prediction']}, margin {case['prediction_margin']:.2f}"
            ),
            true_max,
            low012_limits,
        )

    low012_axes[0].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        ncol=2,
        fontsize=7.5,
    )
    low012_fig.suptitle(
        "Model 1: actual head writes for matched max-0, max-1, and max-2 inputs",
        fontsize=15,
    )
    low012_fig.supxlabel(
        f"Only the center number changes; all panels use actual soft attention and the same "
        f"head scale x {matched_case_display_scale:.5f}",
        fontsize=9,
    )
    low012_fig.savefig(LOW012_OUT, dpi=180, facecolor="white")
    plt.close(low012_fig)

    low0to6_fig = plt.figure(figsize=(22.5, 19.5), constrained_layout=True)
    low0to6_grid = low0to6_fig.add_gridspec(3, 6)
    panel_slots = [
        (0, slice(0, 2)),
        (0, slice(2, 4)),
        (0, slice(4, 6)),
        (1, slice(0, 2)),
        (1, slice(2, 4)),
        (1, slice(4, 6)),
        (2, slice(2, 4)),
    ]
    low0to6_axes = [
        low0to6_fig.add_subplot(low0to6_grid[row, columns], projection="3d")
        for row, columns in panel_slots
    ]
    displayed_piecewise_heads = (
        piecewise_case_head_coordinates_t * piecewise_display_scale
    )
    displayed_piecewise_sums = piecewise_case_sums_t * piecewise_display_scale
    low0to6_points = torch.cat(
        [
            digit_coordinates[:8],
            displayed_piecewise_heads.flatten(0, 1),
            displayed_piecewise_sums,
            torch.zeros(1, 3),
        ]
    )
    low0to6_limits = padded_limits(low0to6_points)

    for true_max, case in enumerate(piecewise_cases):
        if true_max == 0:
            recipe_title = "all heads -> [ANS]"
        elif true_max == 1:
            recipe_title = "H0-H2 -> [ANS]; H3 actual soft mix"
        else:
            recipe_title = f"H0-H2 -> [ANS]; H3 -> {true_max}@5"
        draw_comparison_panel(
            low0to6_axes[true_max],
            displayed_piecewise_heads[true_max],
            displayed_piecewise_sums[true_max],
            piecewise_case_sums_t[true_max],
            (
                f"Max {true_max}: {case['numbers']}\n"
                f"{recipe_title}\n"
                f"3D prediction {case['prediction']}, margin {case['prediction_margin']:.2f}"
            ),
            true_max,
            low0to6_limits,
            tuple(case["selected_unembedding_digits"]),
        )

    head_legend_handles = [
        Line2D([0], [0], color=HEAD_COLORS[idx], lw=2.2, label=label)
        for idx, label in enumerate(HEAD_LABELS)
    ]
    head_legend_handles.append(
        Line2D([0], [0], color="#111827", lw=3.2, label="SUM")
    )
    low0to6_axes[0].legend(
        handles=head_legend_handles,
        loc="upper left",
        frameon=False,
        ncol=2,
        fontsize=7.5,
    )
    low0to6_fig.suptitle(
        "Model 1: piecewise attention writes for matched maxima 0 through 6",
        fontsize=16,
    )
    low0to6_fig.supxlabel(
        "Max 1 retains H3's actual soft mixture; every other displayed head source is "
        f"one-hot. Shared head scale x {piecewise_display_scale:.5f}.",
        fontsize=10,
    )
    low0to6_fig.savefig(LOW0TO6_OUT, dpi=170, facecolor="white")
    plt.close(low0to6_fig)

    print("head,top_source,top_attention,ans_attention,max_digit_attention,pc1,pc2,pc3")
    for head_idx, row in enumerate(attention_summary):
        coords = head_coordinates_t[head_idx]
        print(
            f"H{head_idx},{row['top_source']},{row['top_attention']:.6f},"
            f"{row['ans_self_attention']:.6f},{row['max_digit_attention']:.6f},"
            f"{float(coords[0]):+.6f},{float(coords[1]):+.6f},{float(coords[2]):+.6f}"
        )
    print(
        f"sum,{float(summed_coordinates[0]):+.6f},{float(summed_coordinates[1]):+.6f},"
        f"{float(summed_coordinates[2]):+.6f}"
    )
    print(f"prediction,{prediction},runner_up,{runner_up},margin,{margin:.6f}")
    print(
        f"counterfactual_prediction,{counterfactual_prediction},"
        f"runner_up,{counterfactual_runner_up},margin,{counterfactual_margin:.6f}"
    )
    print(f"wrote,{OUT}")
    print(f"wrote,{ARROW_OUT}")
    print(f"wrote,{COUNTERFACTUAL_OUT}")
    print(f"wrote,{LOW012_OUT}")
    print(f"wrote,{LOW0TO6_OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
