#!/usr/bin/env python3
"""Plot the fixed H0+H1+H2 ANS-self baseline in digit readout space."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "docs" / "assets" / "model1_max2_lowdim_head_geometry.json"
PNG_OUT = ROOT / "docs" / "assets" / "model1_h012_ans_baseline_geometry.png"
HTML_OUT = ROOT / "docs" / "assets" / "model1_h012_ans_baseline_interactive.html"
JSON_OUT = ROOT / "docs" / "assets" / "model1_h012_ans_baseline_geometry.json"

HEAD_COLORS = {"H0": "#2563eb", "H1": "#f59e0b", "H2": "#16a34a", "B": "#111827"}
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


def vector_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    dot = float(a @ b)
    cosine = dot / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return dot, cosine, angle


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
    source = json.loads(INPUT.read_text())
    piecewise_zero = source["piecewise_low_max_0_to_6"]["cases"][0]
    for head in ("H0", "H1", "H2"):
        if piecewise_zero["recipe"][head] != "ANS self one-hot":
            raise AssertionError(f"{head} is not an ANS-self one-hot source")

    digit_coordinates = np.asarray(source["digit_pc_coordinates"], dtype=float)
    head_coordinates = {
        head: np.asarray(piecewise_zero["head_pc_coordinates"][head], dtype=float)
        for head in ("H0", "H1", "H2")
    }
    baseline = sum(head_coordinates.values())
    digit_norms = np.linalg.norm(digit_coordinates, axis=1)
    baseline_norm = float(np.linalg.norm(baseline))
    dots = digit_coordinates @ baseline
    cosines = dots / (digit_norms * baseline_norm)
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    dot_prediction = int(dots.argmax())
    cosine_prediction = int(cosines.argmax())
    display_scale = (
        0.90
        * float(digit_norms.max())
        / max([float(np.linalg.norm(v)) for v in head_coordinates.values()] + [baseline_norm])
    )
    displayed_heads = {head: vector * display_scale for head, vector in head_coordinates.items()}
    displayed_baseline = baseline * display_scale

    data = {
        "description": (
            "H0, H1, and H2 each use a one-hot ANS-query to ANS-key/value read. "
            "Each 16d value is mapped through its own W_O slice and then through the "
            "top-three centered digit-unembedding PCA basis. B is their 3d sum; H3 is omitted."
        ),
        "source_json": str(INPUT.relative_to(ROOT)),
        "head_pc_coordinates": {
            head: vector.tolist() for head, vector in head_coordinates.items()
        },
        "baseline_pc_coordinates": baseline.tolist(),
        "baseline_norm": baseline_norm,
        "display_scale": display_scale,
        "digit_pc_coordinates": digit_coordinates.tolist(),
        "digit_norms": digit_norms.tolist(),
        "dot_product_by_digit": dots.tolist(),
        "cosine_by_digit": cosines.tolist(),
        "angle_degrees_by_digit": angles.tolist(),
        "dot_product_prediction": dot_prediction,
        "cosine_prediction": cosine_prediction,
    }
    JSON_OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    fig = plt.figure(figsize=(20.5, 6.8), constrained_layout=True)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    cosine_ax = fig.add_subplot(1, 3, 2)
    dot_ax = fig.add_subplot(1, 3, 3)

    ax.plot(
        digit_coordinates[:, 0],
        digit_coordinates[:, 1],
        digit_coordinates[:, 2],
        color="#6b7280",
        linewidth=1.4,
        alpha=0.7,
    )
    for digit, vector in enumerate(digit_coordinates):
        ax.scatter(
            vector[0],
            vector[1],
            vector[2],
            color=DIGIT_COLORS[digit],
            s=34,
            depthshade=False,
        )
        ax.text(
            vector[0],
            vector[1],
            vector[2],
            f" U{digit}",
            color=DIGIT_COLORS[digit],
            fontsize=8,
        )

    head_label_offsets = {
        "H0": np.array([0.02, -0.03, 0.035]),
        "H1": np.array([-0.08, 0.02, 0.04]),
        "H2": np.array([0.02, -0.05, -0.04]),
    }
    for head, vector in displayed_heads.items():
        ax.quiver(
            0.0,
            0.0,
            0.0,
            vector[0],
            vector[1],
            vector[2],
            color=HEAD_COLORS[head],
            linewidth=2.5,
            arrow_length_ratio=0.10,
        )
        label_position = vector + head_label_offsets[head]
        ax.text(
            *label_position,
            head,
            color=HEAD_COLORS[head],
            fontsize=9,
            weight="bold",
        )
    ax.quiver(
        0.0,
        0.0,
        0.0,
        displayed_baseline[0],
        displayed_baseline[1],
        displayed_baseline[2],
        color=HEAD_COLORS["B"],
        linewidth=4.0,
        arrow_length_ratio=0.10,
    )
    baseline_label_position = displayed_baseline + np.array([-0.12, -0.04, -0.055])
    ax.text(
        *baseline_label_position,
        " B = H0+H1+H2",
        color=HEAD_COLORS["B"],
        fontsize=10,
        weight="bold",
    )
    all_points = np.vstack(
        [digit_coordinates, *displayed_heads.values(), displayed_baseline, np.zeros(3)]
    )
    ranges = padded_ranges(all_points)
    ax.set_xlim(ranges[0])
    ax.set_ylim(ranges[1])
    ax.set_zlim(ranges[2])
    ax.set_box_aspect([high - low for low, high in ranges])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"ANS-self baseline in digit space\nheads and B x {display_scale:.6f}")
    ax.view_init(elev=22, azim=-58)
    ax.set_proj_type("ortho")
    ax.grid(alpha=0.2)

    digits = np.arange(10)
    cosine_colors = ["#6b7280"] * 10
    cosine_colors[cosine_prediction] = "#dc2626"
    cosine_ax.bar(digits, cosines, color=cosine_colors, alpha=0.9)
    cosine_ax.axhline(0.0, color="#111827", linewidth=0.8)
    cosine_ax.set_xticks(digits)
    cosine_ax.set_ylim(-1.08, 1.08)
    cosine_ax.set_xlabel("Digit")
    cosine_ax.set_ylabel("cosine(B, U[d])")
    cosine_ax.set_title(
        f"Closest direction: U{cosine_prediction}\n"
        f"cosine {cosines[cosine_prediction]:.4f}, angle {angles[cosine_prediction]:.2f} degrees"
    )
    cosine_ax.grid(axis="y", alpha=0.22)

    dot_colors = ["#6b7280"] * 10
    dot_colors[dot_prediction] = "#0f766e"
    dot_ax.bar(digits, dots, color=dot_colors, alpha=0.9)
    dot_ax.axhline(0.0, color="#111827", linewidth=0.8)
    dot_ax.set_xticks(digits)
    dot_ax.set_xlabel("Digit")
    dot_ax.set_ylabel("B dot U[d]")
    dot_ax.set_title(
        f"Largest dot product: U{dot_prediction}\n"
        f"U{dot_prediction}: {dots[dot_prediction]:.2f}; "
        f"U{cosine_prediction}: {dots[cosine_prediction]:.2f}"
    )
    dot_ax.grid(axis="y", alpha=0.22)

    fig.suptitle("Model 1: fixed H0+H1+H2 ANS-self baseline", fontsize=15)
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=180, facecolor="white")
    plt.close(fig)

    interactive = go.Figure()
    interactive.add_trace(
        go.Scatter3d(
            x=digit_coordinates[:, 0],
            y=digit_coordinates[:, 1],
            z=digit_coordinates[:, 2],
            mode="lines",
            line={"color": "#6b7280", "width": 5},
            hoverinfo="skip",
            name="digit order 0 -> 9",
        )
    )

    def add_interactive_vector(
        label: str,
        displayed: np.ndarray,
        raw: np.ndarray,
        color: str,
        width: int,
        hover_extra: str,
    ) -> None:
        interactive.add_trace(
            go.Scatter3d(
                x=[0.0, displayed[0]],
                y=[0.0, displayed[1]],
                z=[0.0, displayed[2]],
                mode="lines+markers+text",
                line={"color": color, "width": width},
                marker={"color": color, "size": [2, 7]},
                text=[None, label],
                textposition="top center",
                textfont={"color": color, "size": 12},
                hovertext=[
                    "origin",
                    (
                        f"<b>{label}</b><br>raw = ({raw[0]:+.6f}, {raw[1]:+.6f}, "
                        f"{raw[2]:+.6f})<br>raw norm = {np.linalg.norm(raw):.6f}"
                        f"{hover_extra}"
                    ),
                ],
                hoverinfo="text",
                name=label,
                legendgroup=label,
            )
        )
        norm = float(np.linalg.norm(displayed))
        direction = displayed / norm
        interactive.add_trace(
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
                legendgroup=label,
                showlegend=False,
            )
        )

    for digit, vector in enumerate(digit_coordinates):
        add_interactive_vector(
            f"U{digit}",
            vector,
            vector,
            DIGIT_COLORS[digit],
            8 if digit in (dot_prediction, cosine_prediction) else 5,
            (
                f"<br>cosine(B, U{digit}) = {cosines[digit]:+.6f}"
                f"<br>angle(B, U{digit}) = {angles[digit]:.3f} degrees"
                f"<br>dot(B, U{digit}) = {dots[digit]:+.6f}"
            ),
        )
    for head, raw in head_coordinates.items():
        add_interactive_vector(
            head,
            displayed_heads[head],
            raw,
            HEAD_COLORS[head],
            7,
            f"<br>display scale = {display_scale:.8f}",
        )
    add_interactive_vector(
        "B = H0+H1+H2",
        displayed_baseline,
        baseline,
        HEAD_COLORS["B"],
        11,
        f"<br>display scale = {display_scale:.8f}",
    )

    interactive.update_layout(
        title={
            "text": (
                "Model 1: H0+H1+H2 ANS-self baseline in digit answer space"
                f"<br><sup>heads and baseline x {display_scale:.8f}; unembeddings unscaled</sup>"
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
        height=840,
        margin={"l": 10, "r": 10, "t": 100, "b": 15},
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
    interactive.write_html(
        HTML_OUT,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )

    print(f"baseline,{baseline.tolist()}")
    print(f"cosine_prediction,{cosine_prediction},cosine,{cosines[cosine_prediction]:.6f}")
    print(f"dot_prediction,{dot_prediction},dot,{dots[dot_prediction]:.6f}")
    print(f"wrote,{PNG_OUT}")
    print(f"wrote,{HTML_OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
