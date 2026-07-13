#!/usr/bin/env python3
"""Build a standalone interactive 3D plot of all digit unembedding vectors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "docs" / "assets" / "model1_max2_lowdim_head_geometry.json"
OUT = ROOT / "docs" / "assets" / "model1_digit_unembedding_3d_interactive.html"
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


def padded_ranges(points: np.ndarray) -> list[list[float]]:
    points = np.vstack([points, np.zeros(3)])
    ranges = []
    for dim in range(3):
        low = min(float(points[:, dim].min()), 0.0)
        high = max(float(points[:, dim].max()), 0.0)
        span = max(high - low, 0.1)
        ranges.append([low - 0.12 * span, high + 0.12 * span])
    return ranges


def main() -> None:
    result = json.loads(INPUT.read_text())
    coordinates = np.asarray(result["digit_pc_coordinates"], dtype=float)
    norms = np.linalg.norm(coordinates, axis=1)
    explained = result["digit_pca_explained_variance"]
    if coordinates.shape != (10, 3) or norms.shape != (10,):
        raise AssertionError(
            f"unexpected coordinate or norm shape: {coordinates.shape}, {norms.shape}"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode="lines",
            line={"color": "#6b7280", "width": 5},
            hoverinfo="skip",
            name="digit order 0 -> 9",
            legendgroup="digit-order",
        )
    )

    for digit, vector in enumerate(coordinates):
        color = DIGIT_COLORS[digit]
        hover = (
            f"<b>U{digit}</b><br>"
            f"PC1 = {vector[0]:+.6f}<br>"
            f"PC2 = {vector[1]:+.6f}<br>"
            f"PC3 = {vector[2]:+.6f}<br>"
            f"norm = {norms[digit]:.6f}"
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, vector[0]],
                y=[0.0, vector[1]],
                z=[0.0, vector[2]],
                mode="lines+markers+text",
                line={"color": color, "width": 6},
                marker={"color": color, "size": [2, 7]},
                text=[None, f"U{digit}"],
                textposition="top center",
                textfont={"color": color, "size": 12},
                hovertext=["origin", hover],
                hoverinfo="text",
                name=f"U{digit}",
                legendgroup=f"U{digit}",
            )
        )
        direction = vector / norms[digit]
        fig.add_trace(
            go.Cone(
                x=[vector[0]],
                y=[vector[1]],
                z=[vector[2]],
                u=[direction[0]],
                v=[direction[1]],
                w=[direction[2]],
                anchor="tip",
                colorscale=[[0.0, color], [1.0, color]],
                showscale=False,
                sizemode="absolute",
                sizeref=0.075,
                hoverinfo="skip",
                legendgroup=f"U{digit}",
                showlegend=False,
            )
        )

    ranges = padded_ranges(coordinates)
    cumulative = sum(explained[:3])
    fig.update_layout(
        title={
            "text": (
                "Model 1: centered digit unembeddings in their top-three PCA basis"
                f"<br><sup>PC1 {explained[0]:.2%}, PC2 {explained[1]:.2%}, "
                f"PC3 {explained[2]:.2%}; cumulative {cumulative:.2%}</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        scene={
            "xaxis": {"title": "PC1", "range": ranges[0], "showspikes": False},
            "yaxis": {"title": "PC2", "range": ranges[1], "showspikes": False},
            "zaxis": {"title": "PC3", "range": ranges[2], "showspikes": False},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 0.95}},
            "bgcolor": "#ffffff",
        },
        height=820,
        margin={"l": 10, "r": 10, "t": 105, "b": 15},
        paper_bgcolor="#ffffff",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
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
                "filename": "model1_digit_unembedding_3d",
                "scale": 2,
            },
        },
    )
    print(f"wrote,{OUT}")


if __name__ == "__main__":
    main()
