#!/usr/bin/env python3
"""Build a standalone interactive 3D comparison for max-3 and max-4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "docs" / "assets" / "model1_max2_lowdim_head_geometry.json"
OUT = ROOT / "docs" / "assets" / "model1_max34_piecewise_interactive.html"

HEAD_COLORS = {
    "H0": "#2563eb",
    "H1": "#f59e0b",
    "H2": "#16a34a",
    "H3": "#dc2626",
    "SUM": "#111827",
}
DIGIT_COLORS = [
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


def vector_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    dot = float(a @ b)
    cosine = dot / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return dot, cosine, angle


def padded_ranges(points: list[np.ndarray]) -> list[list[float]]:
    stacked = np.vstack(points + [np.zeros(3)])
    ranges = []
    for dim in range(3):
        low = min(float(stacked[:, dim].min()), 0.0)
        high = max(float(stacked[:, dim].max()), 0.0)
        span = max(high - low, 0.1)
        ranges.append([low - 0.12 * span, high + 0.12 * span])
    return ranges


def main() -> None:
    result = json.loads(INPUT.read_text())
    digit_coordinates = np.asarray(result["digit_pc_coordinates"], dtype=float)
    piecewise = result["piecewise_low_max_0_to_6"]
    head_scale = float(piecewise["display_scale"])
    cases = {
        int(case["true_max"]): case
        for case in piecewise["cases"]
        if int(case["true_max"]) in (3, 4)
    }
    if set(cases) != {3, 4}:
        raise AssertionError("expected max-3 and max-4 cases in input JSON")

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.03,
        subplot_titles=(
            f"Max 3: prediction 3, margin {cases[3]['prediction_margin']:.2f}",
            f"Max 4: prediction 4, margin {cases[4]['prediction_margin']:.2f}",
        ),
    )

    all_points: list[np.ndarray] = []
    shown_legend_groups: set[str] = set()

    def add_vector(
        *,
        row: int,
        col: int,
        label: str,
        displayed: np.ndarray,
        raw: np.ndarray,
        color: str,
        width: float,
        extra_hover: str,
    ) -> None:
        group = label
        show_legend = group not in shown_legend_groups
        shown_legend_groups.add(group)
        all_points.append(displayed)

        endpoint_hover = (
            f"<b>{label}</b><br>"
            f"displayed = ({displayed[0]:+.5f}, {displayed[1]:+.5f}, {displayed[2]:+.5f})<br>"
            f"raw = ({raw[0]:+.5f}, {raw[1]:+.5f}, {raw[2]:+.5f})<br>"
            f"raw norm = {np.linalg.norm(raw):.5f}{extra_hover}"
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, displayed[0]],
                y=[0.0, displayed[1]],
                z=[0.0, displayed[2]],
                mode="lines+markers+text",
                line={"color": color, "width": width},
                marker={"color": color, "size": [2, 5]},
                text=[None, label],
                textposition="top center",
                textfont={"color": color, "size": 12},
                hovertext=[f"{label}: origin", endpoint_hover],
                hoverinfo="text",
                name=label,
                legendgroup=group,
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )

        norm = float(np.linalg.norm(displayed))
        if norm > 0.0:
            direction = displayed / norm
            fig.add_trace(
                go.Cone(
                    x=[displayed[0]],
                    y=[displayed[1]],
                    z=[displayed[2]],
                    u=[direction[0]],
                    v=[direction[1]],
                    w=[direction[2]],
                    anchor="tip",
                    colorscale=[[0.0, color], [1.0, color]],
                    showscale=False,
                    sizemode="absolute",
                    sizeref=0.07,
                    hoverinfo="skip",
                    legendgroup=group,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    for col, true_max in enumerate((3, 4), start=1):
        case = cases[true_max]
        summed = np.asarray(case["summed_pc_coordinates"], dtype=float)
        for head in ("H0", "H1", "H2", "H3"):
            raw = np.asarray(case["head_pc_coordinates"][head], dtype=float)
            add_vector(
                row=1,
                col=col,
                label=head,
                displayed=raw * head_scale,
                raw=raw,
                color=HEAD_COLORS[head],
                width=7.0,
                extra_hover=f"<br>display scale = {head_scale:.8f}",
            )
        add_vector(
            row=1,
            col=col,
            label="SUM",
            displayed=summed * head_scale,
            raw=summed,
            color=HEAD_COLORS["SUM"],
            width=10.0,
            extra_hover=f"<br>display scale = {head_scale:.8f}",
        )

        for digit in case["selected_unembedding_digits"]:
            vector = digit_coordinates[digit]
            dot, cosine, angle = vector_stats(summed, vector)
            add_vector(
                row=1,
                col=col,
                label=f"U{digit}",
                displayed=vector,
                raw=vector,
                color=DIGIT_COLORS[digit],
                width=9.0 if digit == true_max else 5.0,
                extra_hover=(
                    f"<br>dot(SUM, U{digit}) = {dot:+.5f}"
                    f"<br>cosine(SUM, U{digit}) = {cosine:+.6f}"
                    f"<br>angle(SUM, U{digit}) = {angle:.3f} degrees"
                ),
            )

    axis_ranges = padded_ranges(all_points)
    scene_style = {
        "xaxis": {"title": "PC1", "range": axis_ranges[0], "showspikes": False},
        "yaxis": {"title": "PC2", "range": axis_ranges[1], "showspikes": False},
        "zaxis": {"title": "PC3", "range": axis_ranges[2], "showspikes": False},
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 0.95}},
        "bgcolor": "#ffffff",
    }
    fig.update_layout(
        title={
            "text": (
                "Model 1: interactive max-3 and max-4 head writes"
                f"<br><sup>Equal data-unit scaling; heads and SUM x {head_scale:.8f}; "
                "unembeddings unscaled</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        scene=scene_style,
        scene2=scene_style,
        height=820,
        margin={"l": 10, "r": 10, "t": 110, "b": 15},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "groupclick": "togglegroup",
        },
        hoverlabel={"font_size": 13},
        template="plotly_white",
    )
    fig.write_html(
        OUT,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "model1_max34_piecewise_interactive",
                "scale": 2,
            },
        },
    )
    print(f"wrote,{OUT}")


if __name__ == "__main__":
    main()
