#!/usr/bin/env python3
"""Compare output-matrix and digit-unembedding top-three PC directions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from model1_piecewise_write_animation import load_model


ROOT = Path(__file__).resolve().parents[2]
JSON_OUT = (
    ROOT / "docs" / "assets" / "model1_output_unembedding_pc_alignment.json"
)
HTML_OUT = (
    ROOT / "docs" / "assets" / "model1_output_unembedding_pc_alignment.html"
)
PC_COLORS = ("#2563eb", "#dc2626", "#16a34a")


def fit_top3(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    explained = singular_values.square()
    explained /= explained.sum()
    return vh[:3].T, explained


def align_output_pc_signs(
    unembedding_basis: torch.Tensor, output_basis: torch.Tensor
) -> tuple[torch.Tensor, list[float]]:
    aligned = output_basis.clone()
    signs = []
    for pc_idx in range(3):
        sign = (
            1.0
            if float(unembedding_basis[:, pc_idx] @ aligned[:, pc_idx]) >= 0.0
            else -1.0
        )
        aligned[:, pc_idx] *= sign
        signs.append(sign)
    return aligned, signs


def vector_hover(
    family: str,
    pc_idx: int,
    coordinates: np.ndarray,
    projected_norm: float,
    matched_cosine: float,
    explained_variance: float,
    frame_label: str,
) -> str:
    outside_norm = np.sqrt(max(0.0, 1.0 - projected_norm**2))
    return (
        f"<b>{family} PC{pc_idx + 1}</b><br>"
        f"shown in {frame_label}<br>"
        f"x = {coordinates[0]:+.5f}<br>"
        f"y = {coordinates[1]:+.5f}<br>"
        f"z = {coordinates[2]:+.5f}<br>"
        f"projected norm = {projected_norm:.5f}<br>"
        f"outside displayed subspace = {outside_norm:.5f}<br>"
        f"cosine with matched PC = {matched_cosine:.5f}<br>"
        f"own-matrix variance = {explained_variance:.2%}"
    )


def add_vector_triad(
    fig: go.Figure,
    family: str,
    coordinates: np.ndarray,
    matched_cosines: np.ndarray,
    explained: np.ndarray,
    frame_label: str,
    visible: bool,
) -> list[int]:
    trace_indices = []
    is_unembedding = family == "digit W_U"
    for pc_idx, vector in enumerate(coordinates):
        projected_norm = float(np.linalg.norm(vector))
        fig.add_trace(
            go.Scatter3d(
                x=[0.0, vector[0]],
                y=[0.0, vector[1]],
                z=[0.0, vector[2]],
                mode="lines+markers+text",
                line={
                    "color": PC_COLORS[pc_idx],
                    "width": 8 if is_unembedding else 6,
                    "dash": "solid" if is_unembedding else "dash",
                },
                marker={
                    "color": PC_COLORS[pc_idx],
                    "size": [2, 8],
                    "symbol": "circle" if is_unembedding else "diamond",
                },
                text=[None, f"{'U' if is_unembedding else 'O'} PC{pc_idx + 1}"],
                textposition="top center",
                textfont={"color": PC_COLORS[pc_idx], "size": 12},
                hovertext=[
                    "origin",
                    vector_hover(
                        family=family,
                        pc_idx=pc_idx,
                        coordinates=vector,
                        projected_norm=projected_norm,
                        matched_cosine=float(matched_cosines[pc_idx]),
                        explained_variance=float(explained[pc_idx]),
                        frame_label=frame_label,
                    ),
                ],
                hoverinfo="text",
                name=f"{family} PC{pc_idx + 1}",
                legendgroup=family,
                visible=visible,
            ),
            row=1,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)
    return trace_indices


def add_matched_connectors(
    fig: go.Figure,
    unembedding_coordinates: np.ndarray,
    output_coordinates: np.ndarray,
    matched_cosines: np.ndarray,
    visible: bool,
) -> list[int]:
    trace_indices = []
    for pc_idx in range(3):
        start = unembedding_coordinates[pc_idx]
        end = output_coordinates[pc_idx]
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line={"color": PC_COLORS[pc_idx], "width": 2, "dash": "dot"},
                opacity=0.45,
                hovertext=(
                    f"matched PC{pc_idx + 1}<br>"
                    f"exact 64D cosine = {matched_cosines[pc_idx]:.5f}"
                ),
                hoverinfo="text",
                showlegend=False,
                visible=visible,
            ),
            row=1,
            col=1,
        )
        trace_indices.append(len(fig.data) - 1)
    return trace_indices


def build_figure(data: dict) -> go.Figure:
    overlap = np.asarray(data["comparison"]["pc_cosine_matrix"], dtype=float)
    matched_cosines = np.diag(overlap)
    unembedding_explained = np.asarray(
        data["digit_unembedding_pca"]["explained_variance_by_pc"], dtype=float
    )
    output_explained = np.asarray(
        data["output_matrix_pca"]["explained_variance_by_pc"], dtype=float
    )
    u_frame = data["coordinates"]["digit_unembedding_pc_frame"]
    o_frame = data["coordinates"]["output_matrix_pc_frame"]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "heatmap"}]],
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.08,
    )

    sphere_phi = np.linspace(0.0, np.pi, 30)
    sphere_theta = np.linspace(0.0, 2.0 * np.pi, 45)
    sphere_x = np.outer(np.sin(sphere_phi), np.cos(sphere_theta))
    sphere_y = np.outer(np.sin(sphere_phi), np.sin(sphere_theta))
    sphere_z = np.outer(np.cos(sphere_phi), np.ones_like(sphere_theta))
    fig.add_trace(
        go.Surface(
            x=sphere_x,
            y=sphere_y,
            z=sphere_z,
            surfacecolor=np.zeros_like(sphere_x),
            colorscale=[[0.0, "#cbd5e1"], [1.0, "#cbd5e1"]],
            opacity=0.08,
            showscale=False,
            hoverinfo="skip",
            name="unit sphere",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            marker={"color": "#111827", "size": 4},
            hovertext="origin",
            hoverinfo="text",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    guide_x = [-1.1, 1.1, None, 0.0, 0.0, None, 0.0, 0.0]
    guide_y = [0.0, 0.0, None, -1.1, 1.1, None, 0.0, 0.0]
    guide_z = [0.0, 0.0, None, 0.0, 0.0, None, -1.1, 1.1]
    fig.add_trace(
        go.Scatter3d(
            x=guide_x,
            y=guide_y,
            z=guide_z,
            mode="lines",
            line={"color": "#d1d5db", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    dynamic_trace_indices: dict[str, list[int]] = {"u_frame": [], "o_frame": []}
    for frame_key, frame_data, visible, frame_label in (
        ("u_frame", u_frame, True, "digit W_U PC coordinates"),
        ("o_frame", o_frame, False, "output-matrix PC coordinates"),
    ):
        unembedding_coordinates = np.asarray(
            frame_data["digit_unembedding_pcs"], dtype=float
        )
        output_coordinates = np.asarray(frame_data["output_matrix_pcs"], dtype=float)
        dynamic_trace_indices[frame_key].extend(
            add_vector_triad(
                fig,
                "digit W_U",
                unembedding_coordinates,
                matched_cosines,
                unembedding_explained,
                frame_label,
                visible,
            )
        )
        dynamic_trace_indices[frame_key].extend(
            add_vector_triad(
                fig,
                "output W_O",
                output_coordinates,
                matched_cosines,
                output_explained,
                frame_label,
                visible,
            )
        )
        dynamic_trace_indices[frame_key].extend(
            add_matched_connectors(
                fig,
                unembedding_coordinates,
                output_coordinates,
                matched_cosines,
                visible,
            )
        )

    heatmap_text = np.vectorize(lambda value: f"{value:+.3f}")(overlap)
    fig.add_trace(
        go.Heatmap(
            z=overlap,
            x=["W_O PC1", "W_O PC2", "W_O PC3"],
            y=["W_U PC1", "W_U PC2", "W_U PC3"],
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu",
            reversescale=True,
            text=heatmap_text,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorbar={"title": "64D<br>cosine", "len": 0.58},
            hovertemplate=(
                "%{y} vs %{x}<br>exact 64D cosine = %{z:.6f}<extra></extra>"
            ),
            name="PC cosine matrix",
        ),
        row=1,
        col=2,
    )

    static_indices = set(range(len(fig.data))) - set(
        dynamic_trace_indices["u_frame"] + dynamic_trace_indices["o_frame"]
    )

    def visibility(frame_key: str) -> list[bool]:
        selected = set(dynamic_trace_indices[frame_key]) | static_indices
        return [trace_idx in selected for trace_idx in range(len(fig.data))]

    angles = data["comparison"]["principal_angles_degrees"]
    subtitle = (
        f"principal angles: {angles[0]:.2f}°, {angles[1]:.2f}°, {angles[2]:.2f}°"
    )
    title_u = (
        "<b>Top-three W_O and digit-W_U PC directions</b>"
        f"<br><sup>shown in digit W_U coordinates; {subtitle}</sup>"
    )
    title_o = (
        "<b>Top-three W_O and digit-W_U PC directions</b>"
        f"<br><sup>shown in output W_O coordinates; {subtitle}</sup>"
    )
    common_axis = {
        "range": [-1.15, 1.15],
        "showspikes": False,
        "zeroline": False,
        "showbackground": False,
    }
    fig.update_layout(
        title={"text": title_u, "x": 0.5, "xanchor": "center"},
        scene={
            "xaxis": {**common_axis, "title": "W_U PC1 coordinate"},
            "yaxis": {**common_axis, "title": "W_U PC2 coordinate"},
            "zaxis": {**common_axis, "title": "W_U PC3 coordinate"},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.45, "y": -1.55, "z": 1.05}},
            "bgcolor": "#ffffff",
        },
        xaxis={"title": "output-matrix PCs", "side": "bottom"},
        yaxis={"title": "digit-unembedding PCs", "autorange": "reversed"},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.31,
                "xanchor": "center",
                "y": 1.08,
                "yanchor": "top",
                "showactive": True,
                "buttons": [
                    {
                        "label": "View in W_U basis",
                        "method": "update",
                        "args": [
                            {"visible": visibility("u_frame")},
                            {
                                "title.text": title_u,
                                "scene.xaxis.title.text": "W_U PC1 coordinate",
                                "scene.yaxis.title.text": "W_U PC2 coordinate",
                                "scene.zaxis.title.text": "W_U PC3 coordinate",
                            },
                        ],
                    },
                    {
                        "label": "View in W_O basis",
                        "method": "update",
                        "args": [
                            {"visible": visibility("o_frame")},
                            {
                                "title.text": title_o,
                                "scene.xaxis.title.text": "W_O PC1 coordinate",
                                "scene.yaxis.title.text": "W_O PC2 coordinate",
                                "scene.zaxis.title.text": "W_O PC3 coordinate",
                            },
                        ],
                    },
                ],
            }
        ],
        annotations=[
            {
                "text": (
                    "Solid circles: digit W_U PCs. Dashed diamonds: output W_O PCs. "
                    "A shortened arrow has a component outside the displayed 3D subspace."
                ),
                "x": 0.5,
                "y": -0.12,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 12, "color": "#475569"},
            }
        ],
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=790,
        margin={"l": 20, "r": 20, "t": 125, "b": 95},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "center",
            "x": 0.69,
            "groupclick": "togglegroup",
            "font": {"size": 10},
        },
        hoverlabel={"font_size": 13},
    )
    return fig


def build_data(model) -> dict:
    output_matrix = model.layers[0].W_O.weight.detach().cpu().double().T
    digit_unembedding = model.unembed.weight.detach().cpu().double()[:10]
    output_basis_raw, output_explained = fit_top3(output_matrix)
    unembedding_basis, unembedding_explained = fit_top3(digit_unembedding)
    output_basis, signs = align_output_pc_signs(unembedding_basis, output_basis_raw)

    identity = torch.eye(3, dtype=torch.double)
    if not torch.allclose(unembedding_basis.T @ unembedding_basis, identity, atol=1e-10):
        raise AssertionError("digit-unembedding PCs are not orthonormal")
    if not torch.allclose(output_basis.T @ output_basis, identity, atol=1e-10):
        raise AssertionError("output-matrix PCs are not orthonormal")

    overlap = unembedding_basis.T @ output_basis
    overlap_singular_values = torch.linalg.svdvals(overlap)
    principal_angles = torch.rad2deg(
        torch.arccos(overlap_singular_values.clamp(-1.0, 1.0))
    )
    matched_cosines = torch.diagonal(overlap)
    if torch.any(matched_cosines < 0.0):
        raise AssertionError("same-index PC signs were not aligned")

    u_in_u = identity
    o_in_u = overlap.T
    u_in_o = overlap
    o_in_o = identity
    if not torch.allclose(u_in_u @ o_in_u.T, overlap, atol=1e-10):
        raise AssertionError("W_U-frame coordinates do not reproduce PC cosines")
    if not torch.allclose(u_in_o @ o_in_o.T, overlap, atol=1e-10):
        raise AssertionError("W_O-frame coordinates do not reproduce PC cosines")

    chordal_distance = torch.sqrt(torch.sin(torch.deg2rad(principal_angles)).square().sum())
    return {
        "description": (
            "Top-three centered digit-unembedding and stacked output-matrix PC directions. "
            "The exact 64D directions are shown in either PC triad's 3D coordinate frame."
        ),
        "matrices": {
            "digit_unembedding": {
                "definition": "model.unembed.weight[0:10]",
                "shape": list(digit_unembedding.shape),
                "centering": "subtract the mean of the ten digit-token rows",
            },
            "output_matrix": {
                "definition": "O_all = stored PyTorch W_O.weight.T",
                "shape": list(output_matrix.shape),
                "centering": "subtract the mean of the 64 output-direction rows",
            },
        },
        "digit_unembedding_pca": {
            "basis_shape": list(unembedding_basis.shape),
            "explained_variance_by_pc": [
                float(value) for value in unembedding_explained[:3]
            ],
            "top3_cumulative_explained_variance": float(
                unembedding_explained[:3].sum()
            ),
        },
        "output_matrix_pca": {
            "basis_shape": list(output_basis.shape),
            "explained_variance_by_pc": [
                float(value) for value in output_explained[:3]
            ],
            "top3_cumulative_explained_variance": float(output_explained[:3].sum()),
        },
        "comparison": {
            "sign_alignment": (
                "Each output PC sign is chosen so its dot product with the same-index "
                "digit-unembedding PC is nonnegative."
            ),
            "signs_applied_to_output_pcs": signs,
            "pc_cosine_matrix": [
                [float(value) for value in row] for row in overlap
            ],
            "same_index_pc_cosines": [float(value) for value in matched_cosines],
            "principal_angles_degrees": [float(value) for value in principal_angles],
            "cosines_of_principal_angles": [
                float(value) for value in overlap_singular_values
            ],
            "chordal_subspace_distance": float(chordal_distance),
        },
        "coordinates": {
            "digit_unembedding_pc_frame": {
                "digit_unembedding_pcs": [
                    [float(value) for value in row] for row in u_in_u
                ],
                "output_matrix_pcs": [
                    [float(value) for value in row] for row in o_in_u
                ],
            },
            "output_matrix_pc_frame": {
                "digit_unembedding_pcs": [
                    [float(value) for value in row] for row in u_in_o
                ],
                "output_matrix_pcs": [
                    [float(value) for value in row] for row in o_in_o
                ],
            },
        },
        "validation": {
            "both_pc_triads_orthonormal_in_64d": True,
            "coordinate_dot_products_reproduce_exact_64d_pc_cosines": True,
            "plot_projection_warning": (
                "The union of two distinct 3D subspaces can span up to 6D. In each view, "
                "the selected triad is exact and the other triad is orthogonally projected."
            ),
        },
    }


def main() -> None:
    torch.manual_seed(0)
    model, _ = load_model()
    data = build_data(model)
    JSON_OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    figure = build_figure(data)
    figure.write_html(
        HTML_OUT,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "model1_output_unembedding_pc_alignment",
                "scale": 2,
            },
        },
    )

    comparison = data["comparison"]
    print("same_index_pc_cosines," + ",".join(
        f"{value:.9f}" for value in comparison["same_index_pc_cosines"]
    ))
    print("principal_angles_degrees," + ",".join(
        f"{value:.6f}" for value in comparison["principal_angles_degrees"]
    ))
    print(
        "top3_cumulative_variance,"
        f"digit_W_U={data['digit_unembedding_pca']['top3_cumulative_explained_variance']:.9f},"
        f"output_W_O={data['output_matrix_pca']['top3_cumulative_explained_variance']:.9f}"
    )
    print(f"wrote,{JSON_OUT}")
    print(f"wrote,{HTML_OUT}")


if __name__ == "__main__":
    main()
