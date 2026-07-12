#!/usr/bin/env python3
"""Animate the piecewise head-write mechanism in the 3D digit readout space."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from huggingface_hub import hf_hub_download
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "docs" / "assets"
REFERENCE_JSON = ASSET_DIR / "model1_max2_lowdim_head_geometry.json"
JSON_OUT = ASSET_DIR / "model1_piecewise_write_animation.json"
HTML_OUT = ASSET_DIR / "model1_piecewise_write_animation.html"
MP4_OUT = ASSET_DIR / "model1_piecewise_write_animation.mp4"
POSTER_OUT = ASSET_DIR / "model1_piecewise_write_animation_poster.png"

HF_REPO = "andyrdt/04_2026_puzzle_1a"
NUMBER_POSITIONS = [1, 3, 5, 7, 9]
ANS_POSITION = 10
HEAD_NAMES = ["H0", "H1", "H2", "H3"]

COLORS = {
    "digits": "#64748b",
    "target": "#0f766e",
    "baseline": "#374151",
    "H3": "#dc2626",
    "H2": "#16a34a",
    "H0": "#2563eb",
    "sum": "#111827",
    "winner": "#be123c",
    "bar": "#cbd5e1",
}
DIGIT_COLORS = [
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#dc2626",
    "#7c3aed",
    "#a16207",
    "#db2777",
    "#0891b2",
    "#65a30d",
    "#4f46e5",
]


@dataclass(frozen=True)
class RenderState:
    true_max: int
    progress: tuple[float, ...]
    stage: str
    note: str
    frame_name: str


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


def tokenize(true_max: int) -> torch.Tensor:
    numbers = [0, 0, true_max, 0, 0]
    return torch.tensor(
        [[10, numbers[0], 11, numbers[1], 11, numbers[2], 11, numbers[3], 11, numbers[4], 12]],
        dtype=torch.long,
    )


def align_basis_to_existing_coordinates(
    basis: torch.Tensor, centered_digits: torch.Tensor
) -> torch.Tensor:
    """Resolve SVD sign ambiguity against the already-published PCA orientation."""
    reference = np.asarray(
        json.loads(REFERENCE_JSON.read_text())["digit_pc_coordinates"], dtype=np.float64
    )
    aligned = basis.clone()
    coordinates = (centered_digits @ aligned.T).detach().cpu().numpy()
    for component in range(3):
        if float(coordinates[:, component] @ reference[:, component]) < 0.0:
            aligned[component] *= -1.0
    coordinates = (centered_digits @ aligned.T).detach().cpu().numpy()
    if not np.allclose(coordinates, reference, rtol=2e-5, atol=2e-5):
        max_error = float(np.abs(coordinates - reference).max())
        raise AssertionError(f"PCA coordinates do not match published basis: {max_error=}")
    return aligned


def as_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu()]


def score_summary(vector_3d: torch.Tensor, digit_coordinates: torch.Tensor) -> dict:
    scores = (vector_3d @ digit_coordinates.T).detach()
    top2 = torch.topk(scores, 2)
    return {
        "relative_digit_logits": as_list(scores),
        "prediction": int(top2.indices[0]),
        "runner_up": int(top2.indices[1]),
        "margin": float(top2.values[0] - top2.values[1]),
    }


def full_score_summary(vector_64d: torch.Tensor, digit_unembedding: torch.Tensor) -> dict:
    scores = (vector_64d @ digit_unembedding.T).detach()
    top2 = torch.topk(scores, 2)
    return {
        "digit_logits": as_list(scores),
        "prediction": int(top2.indices[0]),
        "runner_up": int(top2.indices[1]),
        "margin": float(top2.values[0] - top2.values[1]),
    }


def compute_data() -> dict:
    torch.manual_seed(0)
    model, config = load_model()
    layer = model.layers[0]
    digit_unembedding = model.unembed.weight.detach()[:10]
    digit_mean = digit_unembedding.mean(dim=0)
    centered_digits = digit_unembedding - digit_mean
    _, singular_values, directions = torch.linalg.svd(centered_digits, full_matrices=False)
    basis = align_basis_to_existing_coordinates(directions[:3], centered_digits)
    digit_coordinates = centered_digits @ basis.T
    explained = singular_values.square() / singular_values.square().sum()

    w_o = layer.W_O.weight.detach()
    output_matrices = torch.stack(
        [
            w_o[:, head_idx * head.d_head : (head_idx + 1) * head.d_head].T
            for head_idx, head in enumerate(layer.heads)
        ]
    )
    reduced_outputs = output_matrices @ basis.T
    if output_matrices.shape != (4, 16, 64) or reduced_outputs.shape != (4, 16, 3):
        raise AssertionError("unexpected per-head output matrix shapes")

    reference_tokens = tokenize(0)
    positions = torch.arange(reference_tokens.shape[1]).unsqueeze(0)
    reference_resid = model.tok_embed(reference_tokens) + model.pos_embed(positions)
    self_values = torch.stack(
        [
            (reference_resid @ head.W_V.weight.detach().T)[0, ANS_POSITION]
            for head in layer.heads
        ]
    )
    self_writes_64d = torch.einsum("hd,hdm->hm", self_values, output_matrices)
    self_writes_3d = self_writes_64d @ basis.T
    baseline_64d = self_writes_64d[:3].sum(dim=0)
    baseline_3d = self_writes_3d[:3].sum(dim=0)

    baseline_low = score_summary(baseline_3d, digit_coordinates)
    baseline_full = full_score_summary(baseline_64d, digit_unembedding)
    if baseline_low["prediction"] != 2 or baseline_full["prediction"] != 2:
        raise AssertionError("the fixed H0+H1+H2 baseline must predict digit 2")

    cases = []
    mask = torch.tril(torch.ones(11, 11)).unsqueeze(0)
    for true_max in range(10):
        tokens = tokenize(true_max)
        resid = model.tok_embed(tokens) + model.pos_embed(positions)
        source_values = torch.stack(
            [(resid @ head.W_V.weight.detach().T)[0] for head in layer.heads]
        )
        source_writes_64d = torch.einsum("hpd,hdm->hpm", source_values, output_matrices)
        source_writes_3d = source_writes_64d @ basis.T

        with torch.no_grad():
            actual_logits, _ = model(tokens)
        actual_prediction = int(actual_logits[0, ANS_POSITION, :10].argmax())
        if actual_prediction != true_max:
            raise AssertionError(
                f"actual model failed matched input for max {true_max}: {actual_prediction}"
            )

        corrections = []
        recipe = {"H0": "ANS@10", "H1": "ANS@10", "H2": "ANS@10"}
        m1_attention = None

        if true_max == 0:
            h3_write_64d = source_writes_64d[3, ANS_POSITION]
            h3_source = "ANS@10 one-hot"
            recipe["H3"] = h3_source
        elif true_max == 1:
            with torch.no_grad():
                h3_value, h3_attention = layer.heads[3](resid, mask)
            h3_write_64d = h3_value[0, ANS_POSITION] @ output_matrices[3]
            h3_row = h3_attention[0, ANS_POSITION]
            ans_mass = float(h3_row[ANS_POSITION])
            max_mass = float(h3_row[5])
            other_mass = float(1.0 - ans_mass - max_mass)
            h3_source = (
                f"actual soft row: {100 * ans_mass:.2f}% ANS@10 + "
                f"{100 * max_mass:.2f}% 1@5 + {100 * other_mass:.2f}% other"
            )
            recipe["H3"] = h3_source
            m1_attention = {
                "attention_row": as_list(h3_row),
                "ans_self_mass": ans_mass,
                "unique_max_mass": max_mass,
                "other_mass": other_mass,
            }
        else:
            h3_write_64d = source_writes_64d[3, 5]
            h3_source = f"{true_max}@5 one-hot"
            recipe["H3"] = h3_source

        h3_write_3d = h3_write_64d @ basis.T
        corrections.append(
            {
                "head": "H3",
                "operation": h3_source,
                "label": "H3 reads [ANS]" if true_max == 0 else f"H3 reads {true_max}",
                "vector_3d_tensor": h3_write_3d,
                "vector_64d_tensor": h3_write_64d,
            }
        )

        if true_max >= 7:
            h2_max_64d = source_writes_64d[2, 5]
            delta_h2_64d = h2_max_64d - self_writes_64d[2]
            corrections.append(
                {
                    "head": "H2",
                    "operation": f"replace ANS@10 with {true_max}@5 one-hot",
                    "label": f"H2: [ANS] -> {true_max}",
                    "vector_3d_tensor": delta_h2_64d @ basis.T,
                    "vector_64d_tensor": delta_h2_64d,
                    "from_vector_3d": as_list(self_writes_3d[2]),
                    "to_vector_3d": as_list(source_writes_3d[2, 5]),
                }
            )
            recipe["H2"] = f"{true_max}@5 one-hot"

        if true_max == 9:
            h0_max_64d = source_writes_64d[0, 5]
            delta_h0_64d = h0_max_64d - self_writes_64d[0]
            corrections.append(
                {
                    "head": "H0",
                    "operation": "replace ANS@10 with 9@5 one-hot",
                    "label": "H0: [ANS] -> 9",
                    "vector_3d_tensor": delta_h0_64d @ basis.T,
                    "vector_64d_tensor": delta_h0_64d,
                    "from_vector_3d": as_list(self_writes_3d[0]),
                    "to_vector_3d": as_list(source_writes_3d[0, 5]),
                }
            )
            recipe["H0"] = "9@5 one-hot"

        staged_3d = baseline_3d.clone()
        staged_64d = baseline_64d.clone()
        stages = [
            {
                "stage": "B = H0([ANS]) + H1([ANS]) + H2([ANS])",
                "sum_3d": as_list(staged_3d),
                "low_dimensional": score_summary(staged_3d, digit_coordinates),
                "full_64d": full_score_summary(staged_64d, digit_unembedding),
            }
        ]
        for correction in corrections:
            staged_3d = staged_3d + correction["vector_3d_tensor"]
            staged_64d = staged_64d + correction["vector_64d_tensor"]
            stages.append(
                {
                    "stage": correction["label"],
                    "sum_3d": as_list(staged_3d),
                    "low_dimensional": score_summary(staged_3d, digit_coordinates),
                    "full_64d": full_score_summary(staged_64d, digit_unembedding),
                }
            )

        chosen_writes_64d = [self_writes_64d[0], self_writes_64d[1], self_writes_64d[2], h3_write_64d]
        if true_max >= 7:
            chosen_writes_64d[2] = source_writes_64d[2, 5]
        if true_max == 9:
            chosen_writes_64d[0] = source_writes_64d[0, 5]
        direct_final_64d = torch.stack(chosen_writes_64d).sum(dim=0)
        direct_final_3d = direct_final_64d @ basis.T
        if not torch.allclose(staged_64d, direct_final_64d, rtol=1e-5, atol=2e-5):
            raise AssertionError(f"replacement-delta bookkeeping failed for max {true_max}")
        if not torch.allclose(staged_3d, direct_final_3d, rtol=1e-5, atol=2e-5):
            raise AssertionError(f"3D component sum failed for max {true_max}")

        final_low = score_summary(staged_3d, digit_coordinates)
        final_full = full_score_summary(staged_64d, digit_unembedding)
        if final_low["prediction"] != true_max or final_full["prediction"] != true_max:
            raise AssertionError(
                f"forced recipe failed for max {true_max}: "
                f"3D={final_low['prediction']}, 64D={final_full['prediction']}"
            )

        serializable_corrections = []
        for correction in corrections:
            row = {
                key: value
                for key, value in correction.items()
                if not key.endswith("_tensor")
            }
            row["vector_3d"] = as_list(correction["vector_3d_tensor"])
            row["vector_64d"] = as_list(correction["vector_64d_tensor"])
            serializable_corrections.append(row)

        cases.append(
            {
                "true_max": true_max,
                "numbers": [0, 0, true_max, 0, 0],
                "tokens": tokens.squeeze(0).tolist(),
                "recipe": recipe,
                "m1_actual_h3_attention": m1_attention,
                "corrections": serializable_corrections,
                "stages": stages,
                "final": {
                    "sum_3d": as_list(staged_3d),
                    "sum_64d": as_list(staged_64d),
                    "low_dimensional": final_low,
                    "full_64d": final_full,
                    "actual_model_prediction": actual_prediction,
                },
            }
        )

    all_vectors = [baseline_3d]
    for case in cases:
        all_vectors.append(torch.tensor(case["final"]["sum_3d"]))
        for correction in case["corrections"]:
            all_vectors.append(torch.tensor(correction["vector_3d"]))
    max_write_norm = max(float(vector.detach().norm()) for vector in all_vectors)
    max_digit_norm = float(digit_coordinates.norm(dim=1).max())
    display_scale = 0.90 * max_digit_norm / max_write_norm

    high_stage_predictions = {
        str(case["true_max"]): [
            stage["low_dimensional"]["prediction"] for stage in case["stages"]
        ]
        for case in cases
        if case["true_max"] >= 7
    }
    expected_high_stages = {"7": [2, 3, 7], "8": [2, 4, 8], "9": [2, 4, 8, 9]}
    if high_stage_predictions != expected_high_stages:
        raise AssertionError(
            f"unexpected high-digit stage sequence: {high_stage_predictions}"
        )

    return {
        "description": (
            "Exact staged head-write recipes for matched inputs [0,0,m,0,0]. "
            "B contains H0/H1/H2 ANS-self writes. H3 is then added; for maxima 7-9, "
            "H2's source is replaced; for max 9, H0's source is also replaced. Replacement "
            "vectors are represented as max-write minus the self-write already included in B."
        ),
        "hf_repo": HF_REPO,
        "model_config": config,
        "tensor_path": {
            "value": "attention_h[ANS,:] @ (residual @ W_V_h.T), shape 1x16",
            "head_write": "value_h @ O_h, shape 1x64",
            "projected_head_write": "value_h @ O_h @ P3.T, shape 1x3",
            "relative_digit_logits": "sum_h(projected_head_write_h) @ U3.T, shape 1x10",
        },
        "pca": {
            "basis_shape": list(basis.shape),
            "digit_coordinates_shape": list(digit_coordinates.shape),
            "top3_explained_variance": float(explained[:3].sum()),
            "digit_coordinates": [[float(value) for value in row] for row in digit_coordinates],
            "digit_norms": as_list(digit_coordinates.norm(dim=1)),
        },
        "display_scale_for_head_writes": display_scale,
        "baseline": {
            "definition": "H0([ANS]) + H1([ANS]) + H2([ANS])",
            "head_self_writes_3d": {
                f"H{head}": as_list(self_writes_3d[head]) for head in range(3)
            },
            "sum_3d": as_list(baseline_3d),
            "sum_64d": as_list(baseline_64d),
            "low_dimensional": baseline_low,
            "full_64d": baseline_full,
        },
        "cases": cases,
        "validation": {
            "all_ten_final_3d_predictions_correct": True,
            "all_ten_final_64d_head_sum_predictions_correct": True,
            "all_ten_actual_model_predictions_correct": True,
            "all_replacement_deltas_match_direct_head_selection": True,
            "high_stage_3d_predictions": high_stage_predictions,
            "interpolation_warning": (
                "Only correction endpoints are verified attention interventions. "
                "Intermediate lambda-scaled vectors are an explanatory animation."
            ),
        },
    }


def correction_arrays(case: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    vectors_3d = [np.asarray(row["vector_3d"], dtype=float) for row in case["corrections"]]
    vectors_64d = [np.asarray(row["vector_64d"], dtype=float) for row in case["corrections"]]
    return vectors_3d, vectors_64d


def state_vectors(data: dict, state: RenderState) -> dict:
    case = data["cases"][state.true_max]
    vectors_3d, vectors_64d = correction_arrays(case)
    baseline_3d = np.asarray(data["baseline"]["sum_3d"], dtype=float)
    baseline_64d = np.asarray(data["baseline"]["sum_64d"], dtype=float)
    digit_coordinates = np.asarray(data["pca"]["digit_coordinates"], dtype=float)
    progress = np.asarray(state.progress, dtype=float)
    current_3d = baseline_3d + sum(
        (amount * vector for amount, vector in zip(progress, vectors_3d)),
        start=np.zeros(3),
    )
    current_64d = baseline_64d + sum(
        (amount * vector for amount, vector in zip(progress, vectors_64d)),
        start=np.zeros(64),
    )
    scores = current_3d @ digit_coordinates.T
    return {
        "case": case,
        "vectors_3d": vectors_3d,
        "vectors_64d": vectors_64d,
        "baseline_3d": baseline_3d,
        "baseline_64d": baseline_64d,
        "current_3d": current_3d,
        "current_64d": current_64d,
        "scores": scores,
        "prediction": int(scores.argmax()),
    }


def build_interactive_states(data: dict) -> list[RenderState]:
    states = []
    for case in data["cases"]:
        true_max = case["true_max"]
        n_corrections = len(case["corrections"])
        states.append(
            RenderState(
                true_max,
                tuple(0.0 for _ in range(n_corrections)),
                "Start from fixed baseline B",
                "B contains H0/H1/H2 reading [ANS]",
                f"max-{true_max}-baseline",
            )
        )
        progress = [0.0] * n_corrections
        for correction_idx, correction in enumerate(case["corrections"]):
            for step, amount in enumerate(np.linspace(0.125, 1.0, 8), start=1):
                frame_progress = progress.copy()
                frame_progress[correction_idx] = float(amount)
                states.append(
                    RenderState(
                        true_max,
                        tuple(frame_progress),
                        f"{correction['label']}: lambda={amount:.3f}",
                        "Intermediate lambda is illustrative; lambda=1 is the exact intervention",
                        f"max-{true_max}-{correction['head'].lower()}-{step:02d}",
                    )
                )
            progress[correction_idx] = 1.0
        states.append(
            RenderState(
                true_max,
                tuple(1.0 for _ in range(n_corrections)),
                "Verified endpoint",
                "Exact attention-source recipe; the 3D and full 64D readouts both predict the target",
                f"max-{true_max}-final",
            )
        )
    return states


def vector_line_3d(start: np.ndarray, end: np.ndarray, color: str, width: int, name: str):
    visible = bool(np.linalg.norm(end - start) > 1e-10)
    return go.Scatter3d(
        x=[start[0], end[0]] if visible else [],
        y=[start[1], end[1]] if visible else [],
        z=[start[2], end[2]] if visible else [],
        mode="lines",
        line={"color": color, "width": width},
        name=name,
        hoverinfo="skip",
        visible=visible,
        showlegend=False,
    )


def vector_cone_3d(start: np.ndarray, end: np.ndarray, color: str, name: str):
    delta = end - start
    visible = bool(np.linalg.norm(delta) > 1e-10)
    return go.Cone(
        x=[end[0]] if visible else [],
        y=[end[1]] if visible else [],
        z=[end[2]] if visible else [],
        u=[delta[0]] if visible else [],
        v=[delta[1]] if visible else [],
        w=[delta[2]] if visible else [],
        anchor="tip",
        sizemode="absolute",
        sizeref=0.105,
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        name=name,
        hoverinfo="skip",
        visible=visible,
        showlegend=False,
    )


def plotly_dynamic_traces(data: dict, state: RenderState) -> list:
    values = state_vectors(data, state)
    scale = float(data["display_scale_for_head_writes"])
    baseline = values["baseline_3d"] * scale
    current = values["current_3d"] * scale
    target = np.asarray(data["pca"]["digit_coordinates"], dtype=float)[state.true_max]

    traces = [
        go.Scatter3d(
            x=[target[0]],
            y=[target[1]],
            z=[target[2]],
            mode="markers",
            marker={"size": 12, "color": COLORS["target"], "symbol": "diamond"},
            name=f"target U{state.true_max}",
            hovertemplate=f"<b>target U{state.true_max}</b><extra></extra>",
            showlegend=False,
        ),
        vector_line_3d(np.zeros(3), baseline, COLORS["baseline"], 7, "B"),
        vector_cone_3d(np.zeros(3), baseline, COLORS["baseline"], "B"),
    ]

    correction_by_head = {row["head"]: (idx, row) for idx, row in enumerate(values["case"]["corrections"])}
    completed = values["baseline_3d"].copy()
    for head in ("H3", "H2", "H0"):
        if head in correction_by_head:
            idx, correction = correction_by_head[head]
            start = completed * scale
            end_raw = completed + state.progress[idx] * values["vectors_3d"][idx]
            end = end_raw * scale
            completed = end_raw
            traces.extend(
                [
                    vector_line_3d(start, end, COLORS[head], 8, correction["label"]),
                    vector_cone_3d(start, end, COLORS[head], correction["label"]),
                ]
            )
        else:
            traces.extend(
                [
                    vector_line_3d(np.zeros(3), np.zeros(3), COLORS[head], 8, head),
                    vector_cone_3d(np.zeros(3), np.zeros(3), COLORS[head], head),
                ]
            )

    traces.extend(
        [
            vector_line_3d(np.zeros(3), current, COLORS["sum"], 11, "current sum"),
            vector_cone_3d(np.zeros(3), current, COLORS["sum"], "current sum"),
            go.Scatter3d(
                x=[current[0]],
                y=[current[1]],
                z=[current[2]],
                mode="markers+text",
                marker={"size": 8, "color": COLORS["sum"], "symbol": "diamond"},
                text=["z"],
                textposition="top center",
                textfont={"size": 13, "color": COLORS["sum"]},
                hovertemplate=(
                    f"<b>current z</b><br>raw 3D: "
                    f"({values['current_3d'][0]:+.3f}, {values['current_3d'][1]:+.3f}, "
                    f"{values['current_3d'][2]:+.3f})<br>display scale: {scale:.7f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
        ]
    )

    bar_colors = [COLORS["bar"]] * 10
    winner = values["prediction"]
    bar_colors[state.true_max] = COLORS["target"]
    if winner != state.true_max:
        bar_colors[winner] = COLORS["winner"]
    traces.append(
        go.Bar(
            x=list(range(10)),
            y=values["scores"],
            marker_color=bar_colors,
            text=[f"{score:.1f}" for score in values["scores"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="digit %{x}<br>relative logit %{y:.4f}<extra></extra>",
            showlegend=False,
        )
    )
    return traces


def plotly_title(data: dict, state: RenderState) -> str:
    prediction = state_vectors(data, state)["prediction"]
    status = "target wins" if prediction == state.true_max else f"current winner: {prediction}"
    basis_prefix = (
        f"{data['basis_label']} | " if data.get("basis_label") else ""
    )
    return (
        f"<b>{basis_prefix}piecewise writes | target max {state.true_max}</b>"
        f"<br><span style='font-size:15px'>{state.stage} | {status}</span>"
        f"<br><span style='font-size:12px;color:#64748b'>{state.note}</span>"
    )


def scene_ranges(data: dict) -> list[list[float]]:
    scale = float(data["display_scale_for_head_writes"])
    points = [np.zeros(3)]
    points.extend(np.asarray(data["pca"]["digit_coordinates"], dtype=float))
    points.append(np.asarray(data["baseline"]["sum_3d"], dtype=float) * scale)
    for case in data["cases"]:
        points.append(np.asarray(case["final"]["sum_3d"], dtype=float) * scale)
        running = np.asarray(data["baseline"]["sum_3d"], dtype=float).copy()
        for correction in case["corrections"]:
            running += np.asarray(correction["vector_3d"], dtype=float)
            points.append(running * scale)
    array = np.asarray(points)
    ranges = []
    for axis in range(3):
        low = float(array[:, axis].min())
        high = float(array[:, axis].max())
        span = max(high - low, 0.1)
        ranges.append([low - 0.12 * span, high + 0.12 * span])
    return ranges


def all_interactive_scores(data: dict, states: list[RenderState]) -> np.ndarray:
    return np.concatenate([state_vectors(data, state)["scores"] for state in states])


def render_interactive(data: dict, output_path: Path = HTML_OUT) -> None:
    states = build_interactive_states(data)
    digit_coordinates = np.asarray(data["pca"]["digit_coordinates"], dtype=float)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.045,
    )

    ray_x, ray_y, ray_z = [], [], []
    for vector in digit_coordinates:
        ray_x.extend([0.0, vector[0], None])
        ray_y.extend([0.0, vector[1], None])
        ray_z.extend([0.0, vector[2], None])
    fig.add_trace(
        go.Scatter3d(
            x=ray_x,
            y=ray_y,
            z=ray_z,
            mode="lines",
            line={"color": "#cbd5e1", "width": 3},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=digit_coordinates[:, 0],
            y=digit_coordinates[:, 1],
            z=digit_coordinates[:, 2],
            mode="lines+markers+text",
            line={"color": COLORS["digits"], "width": 3},
            marker={"size": 6, "color": DIGIT_COLORS},
            text=[f"U{digit}" for digit in range(10)],
            textposition="top center",
            customdata=np.linalg.norm(digit_coordinates, axis=1),
            hovertemplate=(
                "<b>%{text}</b><br>PC1 %{x:+.5f}<br>PC2 %{y:+.5f}<br>"
                "PC3 %{z:+.5f}<br>norm %{customdata:.5f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    dynamic_indices = []
    initial_traces = plotly_dynamic_traces(data, states[0])
    for trace_idx, trace in enumerate(initial_traces):
        col = 2 if trace_idx == len(initial_traces) - 1 else 1
        fig.add_trace(trace, row=1, col=col)
        dynamic_indices.append(len(fig.data) - 1)

    frames = []
    for state in states:
        frames.append(
            go.Frame(
                name=state.frame_name,
                data=plotly_dynamic_traces(data, state),
                traces=dynamic_indices,
                layout=go.Layout(title={"text": plotly_title(data, state)}),
            )
        )
    fig.frames = frames

    ranges = scene_ranges(data)
    score_values = all_interactive_scores(data, states)
    score_span = float(score_values.max() - score_values.min())
    score_range = [
        float(score_values.min() - 0.10 * score_span),
        float(score_values.max() + 0.16 * score_span),
    ]
    final_frame_names = [f"max-{true_max}-final" for true_max in range(10)]
    slider_steps = [
        {
            "label": str(true_max),
            "method": "animate",
            "args": [
                [frame_name],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for true_max, frame_name in enumerate(final_frame_names)
    ]
    fig.update_layout(
        title={"text": plotly_title(data, states[0]), "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        height=860,
        margin={"l": 12, "r": 18, "t": 112, "b": 120},
        font={"family": "Inter, Arial, sans-serif", "size": 13, "color": "#1f2937"},
        scene={
            "xaxis": {"title": "PC1", "range": ranges[0]},
            "yaxis": {"title": "PC2", "range": ranges[1]},
            "zaxis": {"title": "PC3", "range": ranges[2]},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": -1.65, "z": 1.05}},
        },
        xaxis={"title": "digit", "tickmode": "linear", "dtick": 1, "range": [-0.6, 9.6]},
        yaxis={"title": "relative logit z . U3[d]", "range": score_range, "zeroline": True},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.11,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play all stages",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": False,
                                "frame": {"duration": 165, "redraw": True},
                                "transition": {"duration": 70},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Verified endpoint for max: ", "font": {"size": 14}},
                "pad": {"t": 42},
                "x": 0.25,
                "len": 0.73,
                "steps": slider_steps,
            }
        ],
        annotations=[
            {
                "text": (
                    data.get(
                        "interactive_note",
                        "Head and sum arrows share one positive display scale; digit unembeddings are unscaled. "
                        "The bars use the raw, unscaled 3D dot products.",
                    )
                ),
                "x": 0.5,
                "y": -0.17,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 11, "color": "#64748b"},
            }
        ],
    )
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )


def build_video_states(data: dict) -> list[RenderState]:
    states = [
        RenderState(0, (0.0,), "The fixed baseline B", "H0/H1/H2 all read [ANS]", "intro")
        for _ in range(12)
    ]
    for case in data["cases"]:
        true_max = case["true_max"]
        n_corrections = len(case["corrections"])
        zeros = tuple(0.0 for _ in range(n_corrections))
        states.extend(
            RenderState(
                true_max,
                zeros,
                f"Target max {true_max}: start at B",
                "Current dot-product winner is shown on the right",
                f"max-{true_max}-baseline",
            )
            for _ in range(8)
        )
        progress = [0.0] * n_corrections
        for correction_idx, correction in enumerate(case["corrections"]):
            for amount in np.linspace(1.0 / 18.0, 1.0, 18):
                frame_progress = progress.copy()
                frame_progress[correction_idx] = float(amount)
                states.append(
                    RenderState(
                        true_max,
                        tuple(frame_progress),
                        f"Target max {true_max}: {correction['label']}",
                        f"Illustrative B + lambda*x, lambda={amount:.2f}; endpoint is exact",
                        f"max-{true_max}-{correction_idx}-{amount:.3f}",
                    )
                )
            progress[correction_idx] = 1.0
        final_progress = tuple(1.0 for _ in range(n_corrections))
        states.extend(
            RenderState(
                true_max,
                final_progress,
                f"Target max {true_max}: verified endpoint",
                "Exact recipe predicts the target in both 3D and full 64D",
                f"max-{true_max}-final",
            )
            for _ in range(12)
        )
    states.extend(
        RenderState(
            9,
            (1.0, 1.0, 1.0),
            "One fixed baseline, piecewise routed corrections",
            "H3 handles the low range; H2 and then H0 cross the high-digit boundaries",
            "outro",
        )
        for _ in range(24)
    )
    if len(states) != 488:
        raise AssertionError(f"unexpected video frame count: {len(states)}")
    return states


def matplotlib_limits(data: dict) -> tuple[list[list[float]], tuple[float, float]]:
    ranges = scene_ranges(data)
    states = build_interactive_states(data)
    scores = all_interactive_scores(data, states)
    span = float(scores.max() - scores.min())
    return ranges, (float(scores.min() - 0.10 * span), float(scores.max() + 0.16 * span))


def draw_matplotlib_state(
    fig: plt.Figure,
    geometry_ax,
    score_ax,
    data: dict,
    state: RenderState,
    ranges: list[list[float]],
    score_range: tuple[float, float],
) -> None:
    for text_artist in list(fig.texts):
        text_artist.remove()
    geometry_ax.clear()
    score_ax.clear()
    values = state_vectors(data, state)
    scale = float(data["display_scale_for_head_writes"])
    digit_coordinates = np.asarray(data["pca"]["digit_coordinates"], dtype=float)
    baseline = values["baseline_3d"] * scale
    current = values["current_3d"] * scale

    for digit, vector in enumerate(digit_coordinates):
        geometry_ax.plot(
            [0.0, vector[0]],
            [0.0, vector[1]],
            [0.0, vector[2]],
            color="#d1d5db",
            linewidth=1.0,
            zorder=1,
        )
        geometry_ax.scatter(
            *vector,
            s=56 if digit == state.true_max else 30,
            color=COLORS["target"] if digit == state.true_max else DIGIT_COLORS[digit],
            edgecolor="white",
            linewidth=0.8,
            depthshade=False,
            zorder=3,
        )
        geometry_ax.text(
            *vector,
            f" U{digit}",
            color=COLORS["target"] if digit == state.true_max else DIGIT_COLORS[digit],
            fontsize=9 if digit == state.true_max else 8,
            weight="bold" if digit == state.true_max else "normal",
        )
    geometry_ax.plot(
        digit_coordinates[:, 0],
        digit_coordinates[:, 1],
        digit_coordinates[:, 2],
        color="#94a3b8",
        linewidth=1.3,
        alpha=0.7,
    )

    def draw_arrow(start: np.ndarray, end: np.ndarray, color: str, label: str, width: float) -> None:
        delta = end - start
        if float(np.linalg.norm(delta)) < 1e-10:
            return
        geometry_ax.quiver(
            start[0],
            start[1],
            start[2],
            delta[0],
            delta[1],
            delta[2],
            color=color,
            linewidth=width,
            arrow_length_ratio=0.10,
            zorder=5,
        )
        geometry_ax.text(
            end[0],
            end[1],
            end[2],
            f" {label}",
            color=color,
            fontsize=9,
            weight="bold",
            zorder=6,
        )

    draw_arrow(np.zeros(3), baseline, COLORS["baseline"], "B", 2.4)
    completed = values["baseline_3d"].copy()
    for idx, correction in enumerate(values["case"]["corrections"]):
        start = completed * scale
        completed = completed + state.progress[idx] * values["vectors_3d"][idx]
        end = completed * scale
        draw_arrow(start, end, COLORS[correction["head"]], correction["label"], 2.8)
    draw_arrow(np.zeros(3), current, COLORS["sum"], "z", 4.2)

    for axis, limits in zip((geometry_ax.set_xlim, geometry_ax.set_ylim, geometry_ax.set_zlim), ranges):
        axis(limits)
    geometry_ax.set_box_aspect([limits[1] - limits[0] for limits in ranges])
    geometry_ax.set_xlabel("PC1", labelpad=8)
    geometry_ax.set_ylabel("PC2", labelpad=8)
    geometry_ax.set_zlabel("PC3", labelpad=8)
    geometry_ax.view_init(elev=22, azim=-58)
    geometry_ax.set_proj_type("ortho")
    geometry_ax.grid(alpha=0.18)
    geometry_ax.set_title(
        f"Head-to-tail writes in digit PCA space\n"
        f"head/sum display scale = {scale:.7f}; U0-U9 unscaled",
        fontsize=13,
        pad=16,
    )

    winner = values["prediction"]
    colors = [COLORS["bar"]] * 10
    colors[state.true_max] = COLORS["target"]
    if winner != state.true_max:
        colors[winner] = COLORS["winner"]
    bars = score_ax.bar(np.arange(10), values["scores"], color=colors, width=0.76)
    score_ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    score_ax.set_xlim(-0.6, 9.6)
    score_ax.set_ylim(score_range)
    score_ax.set_xticks(np.arange(10))
    score_ax.set_xlabel("Digit")
    score_ax.set_ylabel("Relative logit: z dot U3[d]")
    score_ax.axhline(0.0, color="#475569", linewidth=0.9)
    score_ax.grid(axis="y", alpha=0.22)
    score_ax.set_title(
        f"Current winner: {winner} | target: {state.true_max}\n"
        f"{'TARGET WINS' if winner == state.true_max else 'correction still in progress'}",
        color=COLORS["target"] if winner == state.true_max else COLORS["winner"],
        fontsize=14,
        pad=16,
    )

    fig.suptitle(state.stage, fontsize=20, weight="bold", y=0.975)
    fig.text(0.5, 0.925, state.note, ha="center", va="center", fontsize=12, color="#475569")
    fig.text(
        0.5,
        0.025,
        (
            "Only the lambda=1 endpoints are exact attention interventions. "
            "For H2/H0, the displayed delta replaces the [ANS] write already included in B."
        ),
        ha="center",
        fontsize=10.5,
        color="#64748b",
    )


def render_poster(data: dict) -> None:
    ranges, score_range = matplotlib_limits(data)
    state = RenderState(
        9,
        (1.0, 1.0, 1.0),
        "Piecewise attention routing writes the answer into a 3D readout",
        "Max 9: B + H3(9) + [H2: ANS -> 9] + [H0: ANS -> 9]",
        "poster",
    )
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    geometry_ax = fig.add_subplot(1, 2, 1, projection="3d")
    score_ax = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.035, right=0.975, top=0.88, bottom=0.09, wspace=0.14)
    draw_matplotlib_state(fig, geometry_ax, score_ax, data, state, ranges, score_range)
    fig.savefig(POSTER_OUT, dpi=100, facecolor="white")
    plt.close(fig)


def render_video(data: dict) -> None:
    ranges, score_range = matplotlib_limits(data)
    states = build_video_states(data)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    geometry_ax = fig.add_subplot(1, 2, 1, projection="3d")
    score_ax = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.035, right=0.975, top=0.88, bottom=0.09, wspace=0.14)
    writer = animation.FFMpegWriter(
        fps=24,
        codec="libx264",
        bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={
            "title": "Model 1 piecewise low-dimensional head writes",
            "artist": "April 2026 Max of List result book",
        },
    )
    last_signature = None
    with writer.saving(fig, MP4_OUT, dpi=100):
        for frame_idx, state in enumerate(states, start=1):
            signature = (state.true_max, state.progress, state.stage, state.note)
            if signature != last_signature:
                draw_matplotlib_state(
                    fig, geometry_ax, score_ax, data, state, ranges, score_range
                )
                last_signature = signature
            writer.grab_frame(facecolor="white")
            if frame_idx % 50 == 0 or frame_idx == len(states):
                print(f"rendered video frames {frame_idx}/{len(states)}", flush=True)
    plt.close(fig)


def print_summary(data: dict) -> None:
    print("max,staged_3d_predictions,final_3d,final_64d,actual")
    for case in data["cases"]:
        stages = "->".join(
            str(stage["low_dimensional"]["prediction"]) for stage in case["stages"]
        )
        print(
            f"{case['true_max']},{stages},"
            f"{case['final']['low_dimensional']['prediction']},"
            f"{case['final']['full_64d']['prediction']},"
            f"{case['final']['actual_model_prediction']}"
        )
    m1 = data["cases"][1]["m1_actual_h3_attention"]
    print(
        "max1_h3_attention,"
        f"ANS={m1['ans_self_mass']:.6f},max={m1['unique_max_mass']:.6f},"
        f"other={m1['other_mass']:.6f}"
    )
    print(f"display_scale,{data['display_scale_for_head_writes']:.9f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Generate JSON, HTML, and poster without the MP4.",
    )
    args = parser.parse_args()

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    data = compute_data()
    JSON_OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    render_interactive(data)
    render_poster(data)
    if not args.skip_video:
        render_video(data)
    print_summary(data)
    print(f"wrote,{JSON_OUT}")
    print(f"wrote,{HTML_OUT}")
    print(f"wrote,{POSTER_OUT}")
    if not args.skip_video:
        print(f"wrote,{MP4_OUT}")


if __name__ == "__main__":
    main()
