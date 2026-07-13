#!/usr/bin/env python3
"""Visualize direct per-head writes and their sum in output-matrix PCA space."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from model1_output_pca_piecewise_interactive import aligned_output_basis
from model1_piecewise_write_animation import ANS_POSITION, load_model, tokenize


ROOT = Path(__file__).resolve().parents[2]
JSON_OUT = ROOT / "docs" / "assets" / "model1_output_pca_head_contributions_interactive.json"
HTML_OUT = ROOT / "docs" / "assets" / "model1_output_pca_head_contributions_interactive.html"
CORRECTION_JSON = ROOT / "docs" / "assets" / "model1_output_pca_piecewise_interactive.json"

HEAD_COLORS = ("#2563eb", "#f97316", "#16a34a", "#dc2626")
DIGIT_COLORS = (
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
)
VOCAB_LABELS = [str(digit) for digit in range(10)] + ["[BOS]", "[SEP]", "[ANS]", "[EOS]"]


def values_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu()]


def compact_source(source: str) -> str:
    return "soft [ANS]/1" if source.startswith("soft:") else source


def output_matrices(model) -> torch.Tensor:
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
    return matrices


def build_data() -> dict:
    torch.manual_seed(0)
    model, config = load_model()
    layer = model.layers[0]
    basis, basis_metadata = aligned_output_basis(model)
    matrices = output_matrices(model)

    unembedding = model.unembed.weight.detach().cpu().double()
    centered_unembedding = unembedding - unembedding.mean(dim=0, keepdim=True)
    unembedding_3d = centered_unembedding @ basis

    positions = torch.arange(11).unsqueeze(0)
    causal_mask = torch.tril(torch.ones(11, 11)).unsqueeze(0)
    correction_cases = json.loads(CORRECTION_JSON.read_text())["cases"]
    cases = []
    all_writes = []
    endpoint_errors = []

    for target in range(10):
        tokens = tokenize(target)
        with torch.no_grad():
            residual = model.tok_embed(tokens) + model.pos_embed(positions)
            source_values = torch.stack(
                [
                    (residual @ head.W_V.weight.detach().T)[0].double()
                    for head in layer.heads
                ]
            )
            source_writes = torch.einsum("hpd,hdm->hpm", source_values, matrices)

            chosen = [source_writes[head_idx, ANS_POSITION] for head_idx in range(4)]
            sources = ["[ANS]", "[ANS]", "[ANS]", "[ANS]"]
            h3_attention = None

            if target == 1:
                h3_values, h3_pattern = layer.heads[3](residual, causal_mask)
                chosen[3] = h3_values[0, ANS_POSITION].double() @ matrices[3]
                pattern = h3_pattern[0, ANS_POSITION]
                ans_mass = float(pattern[ANS_POSITION])
                target_mass = float(pattern[5])
                sources[3] = f"soft: {100 * ans_mass:.1f}% [ANS] + {100 * target_mass:.1f}% 1"
                h3_attention = {
                    "row": values_list(pattern),
                    "ans_mass": ans_mass,
                    "target_mass": target_mass,
                    "other_mass": float(1.0 - ans_mass - target_mass),
                }
            elif target >= 2:
                chosen[3] = source_writes[3, 5]
                sources[3] = str(target)

            if target >= 7:
                chosen[2] = source_writes[2, 5]
                sources[2] = str(target)

            if target == 9:
                chosen[0] = source_writes[0, 5]
                sources[0] = str(target)

        head_writes_64d = torch.stack(chosen)
        head_writes_3d = head_writes_64d @ basis
        head_sum_64d = head_writes_64d.sum(dim=0)
        head_sum_3d = head_writes_3d.sum(dim=0)
        scores_3d = head_sum_3d @ unembedding_3d.T
        scores_64d = head_sum_64d @ centered_unembedding.T
        top2_3d = torch.topk(scores_3d, 2)
        top2_64d = torch.topk(scores_64d, 2)
        prediction_3d = int(top2_3d.indices[0])
        prediction_64d = int(top2_64d.indices[0])
        if prediction_3d != target or prediction_64d != target:
            raise AssertionError(
                f"direct recipe failed for target {target}: 3D={prediction_3d}, 64D={prediction_64d}"
            )
        if not torch.allclose(head_writes_3d.sum(dim=0), head_sum_3d):
            raise AssertionError(f"projected head sum mismatch for target {target}")
        correction_endpoint = torch.tensor(
            correction_cases[target]["final"]["sum_64d"], dtype=torch.double
        )
        endpoint_error = float((head_sum_64d - correction_endpoint).abs().max())
        endpoint_errors.append(endpoint_error)
        if endpoint_error > 2e-5:
            raise AssertionError(
                f"direct and correction endpoint mismatch for target {target}: {endpoint_error}"
            )

        all_writes.extend(head_writes_3d)
        all_writes.append(head_sum_3d)
        cases.append(
            {
                "target": target,
                "numbers": [0, 0, target, 0, 0],
                "head_sources": {f"H{idx}": source for idx, source in enumerate(sources)},
                "h3_actual_attention_for_target_1": h3_attention,
                "head_writes_3d": [values_list(row) for row in head_writes_3d],
                "head_writes_64d": [values_list(row) for row in head_writes_64d],
                "sum_3d": values_list(head_sum_3d),
                "sum_64d": values_list(head_sum_64d),
                "full_vocab_relative_logits_3d": values_list(scores_3d),
                "prediction_3d": prediction_3d,
                "runner_up_3d": int(top2_3d.indices[1]),
                "margin_3d": float(top2_3d.values[0] - top2_3d.values[1]),
                "prediction_64d": prediction_64d,
                "runner_up_64d": int(top2_64d.indices[1]),
                "margin_64d": float(top2_64d.values[0] - top2_64d.values[1]),
            }
        )

    max_candidate_norm = float(unembedding_3d.norm(dim=1).max())
    max_write_norm = max(float(vector.norm()) for vector in all_writes)
    display_scale = 0.90 * max_candidate_norm / max_write_norm

    return {
        "description": (
            "Direct per-head causal endpoint writes for canonical inputs [0,0,n,0,0], "
            "projected into the top three centered output-matrix PCs. Each Hh arrow is "
            "V_h @ W_O^h; the sum arrow is their vector sum. Target 1 uses H3's measured "
            "soft attention row and all other targets use the verified one-hot recipes."
        ),
        "hf_repo": "andyrdt/04_2026_puzzle_1a",
        "source_correction_interactive": str(CORRECTION_JSON.relative_to(ROOT)),
        "model_config": config,
        "basis": basis_metadata,
        "tensor_path": {
            "head_value": "attention_h[ANS,:] @ (residual @ W_V_h.T), shape 1x16",
            "head_write_64d": "head_value_h @ W_O^h, shape 1x64",
            "head_write_3d": "head_write_64d @ Q_3, shape 1x3",
            "head_sum_3d": "sum_h(head_write_3d_h), shape 1x3",
            "full_vocab_logits_3d": "head_sum_3d @ ((W_U - mean_vocab(W_U)) @ Q_3).T",
        },
        "vocab_labels": VOCAB_LABELS,
        "unembedding_coordinates_3d": [values_list(row) for row in unembedding_3d],
        "display_scale_for_head_and_sum_arrows": display_scale,
        "cases": cases,
        "validation": {
            "all_ten_3d_full_vocab_predictions_correct": True,
            "all_ten_64d_full_vocab_predictions_correct": True,
            "all_projected_head_sums_equal_sum_of_projected_heads": True,
            "basis_fitted_only_to_centered_output_matrix": True,
            "direct_sums_match_correction_interactive_endpoints": True,
            "max_abs_64d_endpoint_difference": max(endpoint_errors),
        },
    }


def vector_traces(
    name: str,
    vector: np.ndarray,
    color: str,
    width: int,
    scale: float,
    showlegend: bool,
) -> list[go.BaseTraceType]:
    end = vector * scale
    line = go.Scatter3d(
        x=[0.0, end[0]],
        y=[0.0, end[1]],
        z=[0.0, end[2]],
        mode="lines",
        line={"color": color, "width": width},
        name=name,
        legendgroup=name,
        showlegend=showlegend,
        hovertemplate=(
            f"<b>{name}</b><br>raw 3D: "
            f"({vector[0]:+.4f}, {vector[1]:+.4f}, {vector[2]:+.4f})"
            f"<br>display scale: {scale:.7f}<extra></extra>"
        ),
    )
    cone = go.Cone(
        x=[end[0]],
        y=[end[1]],
        z=[end[2]],
        u=[end[0]],
        v=[end[1]],
        w=[end[2]],
        anchor="tip",
        sizemode="absolute",
        sizeref=0.065 if name != "sum z" else 0.115,
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        name=name,
        legendgroup=name,
        showlegend=False,
        hoverinfo="skip",
    )
    return [line, cone]


def dynamic_traces(data: dict, case: dict) -> list[go.BaseTraceType]:
    scale = float(data["display_scale_for_head_and_sum_arrows"])
    target = int(case["target"])
    candidate_coordinates = np.asarray(data["unembedding_coordinates_3d"], dtype=float)
    target_coordinate = candidate_coordinates[target]
    traces: list[go.BaseTraceType] = [
        go.Scatter3d(
            x=[target_coordinate[0]],
            y=[target_coordinate[1]],
            z=[target_coordinate[2]],
            mode="markers",
            marker={
                "size": 18,
                "color": "rgba(15, 118, 110, 0.15)",
                "symbol": "circle",
                "line": {"color": "#0f766e", "width": 3},
            },
            name=f"maximum U{target}",
            showlegend=False,
            hovertemplate=f"<b>maximum digit: {target}</b><br>U{target}<extra></extra>",
        )
    ]
    for head_idx, vector in enumerate(case["head_writes_3d"]):
        source = case["head_sources"][f"H{head_idx}"]
        traces.extend(
            vector_traces(
                f"H{head_idx}({compact_source(source)})",
                np.asarray(vector, dtype=float),
                HEAD_COLORS[head_idx],
                3,
                scale,
                True,
            )
        )
    traces.extend(
        vector_traces(
            "sum z",
            np.asarray(case["sum_3d"], dtype=float),
            "#111827",
            9,
            scale,
            True,
        )
    )

    colors = ["#cbd5e1"] * 14
    colors[target] = "#0f766e"
    traces.append(
        go.Bar(
            x=data["vocab_labels"],
            y=case["full_vocab_relative_logits_3d"],
            marker_color=colors,
            text=[f"{score:.1f}" for score in case["full_vocab_relative_logits_3d"]],
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
            hovertemplate="token %{x}<br>relative logit %{y:.5f}<extra></extra>",
        )
    )
    return traces


def title(case: dict) -> str:
    recipe = ", ".join(
        f"{head}: {compact_source(source)}"
        for head, source in case["head_sources"].items()
    )
    return (
        f"<span style='font-size:22px'><b>Maximum digit: {case['target']}</b></span>"
        f"<br><span style='font-size:13px'>Direct per-head writes for "
        f"{case['numbers']}</span>"
        f"<br><span style='font-size:13px'>{recipe}</span>"
        f"<br><span style='font-size:12px;color:#64748b'>"
        f"3D/full-64D winners: {case['prediction_3d']}/{case['prediction_64d']}"
        "</span>"
    )


def scene_ranges(data: dict) -> list[list[float]]:
    scale = float(data["display_scale_for_head_and_sum_arrows"])
    points = [np.zeros(3)]
    points.extend(np.asarray(data["unembedding_coordinates_3d"], dtype=float))
    for case in data["cases"]:
        points.extend(np.asarray(case["head_writes_3d"], dtype=float) * scale)
        points.append(np.asarray(case["sum_3d"], dtype=float) * scale)
    array = np.asarray(points)
    ranges = []
    for axis in range(3):
        low = float(array[:, axis].min())
        high = float(array[:, axis].max())
        span = max(high - low, 0.1)
        ranges.append([low - 0.12 * span, high + 0.12 * span])
    return ranges


def render_interactive(data: dict) -> None:
    candidate_coordinates = np.asarray(data["unembedding_coordinates_3d"], dtype=float)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.67, 0.33],
        horizontal_spacing=0.045,
    )

    candidate_colors = list(DIGIT_COLORS) + ["#64748b"] * 4
    fig.add_trace(
        go.Scatter3d(
            x=candidate_coordinates[:, 0],
            y=candidate_coordinates[:, 1],
            z=candidate_coordinates[:, 2],
            mode="markers+text",
            marker={"size": 7, "color": candidate_colors},
            text=data["vocab_labels"],
            textposition="top center",
            textfont={"size": 10},
            customdata=np.linalg.norm(candidate_coordinates, axis=1),
            hovertemplate=(
                "<b>%{text}</b><br>PC1 %{x:+.5f}<br>PC2 %{y:+.5f}<br>"
                "PC3 %{z:+.5f}<br>norm %{customdata:.5f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    initial = dynamic_traces(data, data["cases"][0])
    dynamic_indices = []
    for trace_idx, trace in enumerate(initial):
        col = 2 if trace_idx == len(initial) - 1 else 1
        fig.add_trace(trace, row=1, col=col)
        dynamic_indices.append(len(fig.data) - 1)

    frames = []
    for case in data["cases"]:
        frames.append(
            go.Frame(
                name=f"target-{case['target']}",
                data=dynamic_traces(data, case),
                traces=dynamic_indices,
                layout=go.Layout(title={"text": title(case)}),
            )
        )
    fig.frames = frames

    score_array = np.asarray(
        [case["full_vocab_relative_logits_3d"] for case in data["cases"]], dtype=float
    )
    score_span = float(score_array.max() - score_array.min())
    ranges = scene_ranges(data)
    slider_steps = [
        {
            "label": str(case["target"]),
            "method": "animate",
            "args": [
                [f"target-{case['target']}"],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for case in data["cases"]
    ]
    fig.update_layout(
        title={"text": title(data["cases"][0]), "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        height=880,
        margin={"l": 12, "r": 18, "t": 170, "b": 118},
        font={"family": "Inter, Arial, sans-serif", "size": 13, "color": "#1f2937"},
        uirevision="output-pca-direct-head-writes",
        scene={
            "xaxis": {"title": "output PC1", "range": ranges[0], "showspikes": False},
            "yaxis": {"title": "output PC2", "range": ranges[1], "showspikes": False},
            "zaxis": {"title": "output PC3", "range": ranges[2], "showspikes": False},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": -1.65, "z": 1.05}},
        },
        xaxis={"title": "full-vocabulary token", "tickangle": -35},
        yaxis={
            "title": "relative logit in output-PC basis",
            "range": [
                float(score_array.min() - 0.08 * score_span),
                float(score_array.max() + 0.16 * score_span),
            ],
            "zeroline": True,
        },
        legend={
            "orientation": "h",
            "x": 0.34,
            "xanchor": "center",
            "y": 0.93,
            "yanchor": "bottom",
            "font": {"size": 10},
        },
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
                        "label": "Play 0-9",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": False,
                                "frame": {"duration": 700, "redraw": True},
                                "transition": {"duration": 180},
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
                "currentvalue": {"prefix": "Maximum digit: ", "font": {"size": 14}},
                "pad": {"t": 42},
                "x": 0.25,
                "len": 0.73,
                "steps": slider_steps,
            }
        ],
        annotations=[
            {
                "text": (
                    "Dots are projected unembedding vectors. Thin colored arrows are "
                    "direct V_h W_O^h writes; the thick black arrow is their sum. "
                    "All arrows share one display scale; bars use unscaled 3D coordinates."
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
        HTML_OUT,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )
    html = HTML_OUT.read_text()
    HTML_OUT.write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n")


def main() -> None:
    data = build_data()
    JSON_OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    render_interactive(data)
    print("target,prediction_3d,prediction_64d,margin_3d,recipe")
    for case in data["cases"]:
        recipe = "; ".join(
            f"{head}->{source}" for head, source in case["head_sources"].items()
        )
        print(
            f"{case['target']},{case['prediction_3d']},{case['prediction_64d']},"
            f"{case['margin_3d']:.6f},{recipe}"
        )
    print(f"wrote,{JSON_OUT}")
    print(f"wrote,{HTML_OUT}")


if __name__ == "__main__":
    main()
