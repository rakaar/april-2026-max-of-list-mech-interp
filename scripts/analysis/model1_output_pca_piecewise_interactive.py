#!/usr/bin/env python3
"""Re-render the piecewise head-write animation in the output-matrix PCA basis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from model1_piecewise_write_animation import load_model, render_interactive


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_JSON = ROOT / "docs" / "assets" / "model1_piecewise_write_animation.json"
ACCURACY_JSON = ROOT / "docs" / "assets" / "model1_output_pca_readout_accuracy.json"
JSON_OUT = ROOT / "docs" / "assets" / "model1_output_pca_piecewise_interactive.json"
HTML_OUT = ROOT / "docs" / "assets" / "model1_output_pca_piecewise_interactive.html"


def score_summary(vector: torch.Tensor, digit_coordinates: torch.Tensor) -> dict:
    scores = vector @ digit_coordinates.T
    top2 = torch.topk(scores, 2)
    return {
        "relative_digit_logits": [float(value) for value in scores],
        "prediction": int(top2.indices[0]),
        "runner_up": int(top2.indices[1]),
        "margin": float(top2.values[0] - top2.values[1]),
    }


def aligned_output_basis(model) -> tuple[torch.Tensor, dict]:
    stored_w_o = model.layers[0].W_O.weight.detach().cpu().double()
    output_matrix = stored_w_o.T
    centered_output = output_matrix - output_matrix.mean(dim=0, keepdim=True)
    _, output_singular_values, output_vh = torch.linalg.svd(
        centered_output, full_matrices=False
    )
    output_basis = output_vh[:3].T

    digit_unembedding = model.unembed.weight.detach().cpu().double()[:10]
    centered_digits = digit_unembedding - digit_unembedding.mean(dim=0, keepdim=True)
    _, _, digit_vh = torch.linalg.svd(centered_digits, full_matrices=False)
    digit_basis = digit_vh[:3].T

    signs = []
    for pc_idx in range(3):
        sign = 1.0 if float(output_basis[:, pc_idx] @ digit_basis[:, pc_idx]) >= 0.0 else -1.0
        output_basis[:, pc_idx] *= sign
        signs.append(sign)

    output_energy = output_singular_values.square()
    output_explained = output_energy / output_energy.sum()
    overlap_singular_values = torch.linalg.svdvals(digit_basis.T @ output_basis)
    principal_angles = torch.rad2deg(
        torch.arccos(overlap_singular_values.clamp(-1.0, 1.0))
    )
    axis_cosines = torch.diagonal(digit_basis.T @ output_basis)
    metadata = {
        "source": "centered rows of O_all = stored PyTorch W_O.weight.T",
        "output_matrix_shape": list(output_matrix.shape),
        "centering": "subtract the mean of the 64 output-direction rows",
        "basis_shape": list(output_basis.shape),
        "sign_alignment": (
            "Each output PC sign is chosen to have nonnegative dot product with the "
            "same-index digit-unembedding PC. Sign changes do not alter scores."
        ),
        "signs_applied": signs,
        "axis_cosine_with_same_index_digit_unembedding_pc": [
            float(value) for value in axis_cosines
        ],
        "principal_angles_to_digit_unembedding_top3_subspace_degrees": [
            float(value) for value in principal_angles
        ],
        "explained_variance_by_pc": [float(value) for value in output_explained[:3]],
        "top3_cumulative_explained_variance": float(output_explained[:3].sum()),
    }
    return output_basis, metadata


def build_data(model) -> dict:
    canonical = json.loads(CANONICAL_JSON.read_text())
    accuracy = json.loads(ACCURACY_JSON.read_text())
    basis, basis_metadata = aligned_output_basis(model)

    digit_unembedding = model.unembed.weight.detach().cpu().double()[:10]
    centered_digits = digit_unembedding - digit_unembedding.mean(dim=0, keepdim=True)
    digit_coordinates = centered_digits @ basis
    baseline_64d = torch.tensor(canonical["baseline"]["sum_64d"], dtype=torch.double)
    baseline_3d = baseline_64d @ basis
    baseline_low = score_summary(baseline_3d, digit_coordinates)

    cases = []
    all_vectors = [baseline_3d]
    stage_predictions = {}
    for canonical_case in canonical["cases"]:
        true_max = canonical_case["true_max"]
        corrections = []
        running_64d = baseline_64d.clone()
        running_3d = baseline_3d.clone()
        stages = [
            {
                "stage": "B = H0([ANS]) + H1([ANS]) + H2([ANS])",
                "sum_3d": [float(value) for value in running_3d],
                "low_dimensional": score_summary(running_3d, digit_coordinates),
            }
        ]

        for canonical_correction in canonical_case["corrections"]:
            vector_64d = torch.tensor(
                canonical_correction["vector_64d"], dtype=torch.double
            )
            vector_3d = vector_64d @ basis
            correction = {
                key: value
                for key, value in canonical_correction.items()
                if key
                not in {
                    "vector_3d",
                    "vector_64d",
                    "from_vector_3d",
                    "to_vector_3d",
                }
            }
            correction["vector_3d"] = [float(value) for value in vector_3d]
            correction["vector_64d"] = [float(value) for value in vector_64d]
            corrections.append(correction)
            running_64d += vector_64d
            running_3d += vector_3d
            all_vectors.append(vector_3d)
            stages.append(
                {
                    "stage": correction["label"],
                    "sum_3d": [float(value) for value in running_3d],
                    "low_dimensional": score_summary(running_3d, digit_coordinates),
                }
            )

        canonical_final = torch.tensor(
            canonical_case["final"]["sum_64d"], dtype=torch.double
        )
        if not torch.allclose(running_64d, canonical_final, rtol=1e-6, atol=3e-5):
            raise AssertionError(f"canonical component sum failed for max {true_max}")
        final_low = score_summary(running_3d, digit_coordinates)
        if final_low["prediction"] != true_max:
            raise AssertionError(
                f"output-PCA endpoint failed for max {true_max}: {final_low['prediction']}"
            )

        expected_stages = [
            stage["low_dimensional"]["prediction"]
            for stage in canonical_case["stages"]
        ]
        observed_stages = [
            stage["low_dimensional"]["prediction"] for stage in stages
        ]
        if observed_stages != expected_stages:
            raise AssertionError(
                f"stage winners changed for max {true_max}: "
                f"output={observed_stages}, unembedding={expected_stages}"
            )
        stage_predictions[str(true_max)] = observed_stages
        all_vectors.append(running_3d)

        cases.append(
            {
                "true_max": true_max,
                "numbers": canonical_case["numbers"],
                "tokens": canonical_case["tokens"],
                "recipe": canonical_case["recipe"],
                "m1_actual_h3_attention": canonical_case["m1_actual_h3_attention"],
                "corrections": corrections,
                "stages": stages,
                "final": {
                    "sum_3d": [float(value) for value in running_3d],
                    "sum_64d": [float(value) for value in running_64d],
                    "low_dimensional": final_low,
                    "full_64d": canonical_case["final"]["full_64d"],
                    "actual_model_prediction": canonical_case["final"][
                        "actual_model_prediction"
                    ],
                },
            }
        )

    max_digit_norm = float(digit_coordinates.norm(dim=1).max())
    max_write_norm = max(float(vector.norm()) for vector in all_vectors)
    display_scale = 0.90 * max_digit_norm / max_write_norm

    output_accuracy = accuracy["evaluation"]["output_pca_readout"]
    if output_accuracy["full_vocabulary"]["3"]["accuracy"] != 1.0:
        raise AssertionError("all-input full-vocabulary output-PCA accuracy is not perfect")

    return {
        "description": (
            "The exact full-64D baseline and piecewise correction vectors from the original "
            "digit-unembedding-PCA animation are projected into the top three PCs of the "
            "centered total output matrix O_all = W_O.T. Digit unembeddings and live dot-product "
            "bars use that same output-derived basis."
        ),
        "basis_label": "Output-matrix PCA",
        "interactive_note": (
            "Coordinates use the top three centered O_all = W_O.T PCs. Head and sum arrows "
            "share one positive display scale; digit unembeddings are unscaled. Bars are raw "
            "3D dot products."
        ),
        "source_canonical_json": str(CANONICAL_JSON.relative_to(ROOT)),
        "source_accuracy_json": str(ACCURACY_JSON.relative_to(ROOT)),
        "basis": basis_metadata,
        "pca": {
            "basis_source": "centered total output matrix",
            "basis_shape": list(basis.shape),
            "top3_explained_variance": basis_metadata[
                "top3_cumulative_explained_variance"
            ],
            "digit_coordinates": [
                [float(value) for value in row] for row in digit_coordinates
            ],
            "digit_norms": [
                float(value) for value in digit_coordinates.norm(dim=1)
            ],
        },
        "display_scale_for_head_writes": display_scale,
        "baseline": {
            "definition": "H0([ANS]) + H1([ANS]) + H2([ANS])",
            "sum_3d": [float(value) for value in baseline_3d],
            "sum_64d": [float(value) for value in baseline_64d],
            "low_dimensional": baseline_low,
            "full_64d": canonical["baseline"]["full_64d"],
        },
        "cases": cases,
        "validation": {
            "all_ten_piecewise_endpoints_correct": True,
            "all_100000_actual_inputs_correct_in_output_top3_basis": True,
            "digit_and_output_basis_completed_stage_winners_identical": True,
            "stage_predictions": stage_predictions,
            "interpolation_warning": (
                "Only correction endpoints are verified attention interventions. "
                "Intermediate lambda-scaled vectors are explanatory interpolation."
            ),
        },
    }


def main() -> None:
    torch.manual_seed(0)
    model, _ = load_model()
    data = build_data(model)
    JSON_OUT.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    render_interactive(data, output_path=HTML_OUT)

    print("max,stage_predictions,final_prediction")
    for case in data["cases"]:
        stages = "->".join(
            str(stage["low_dimensional"]["prediction"])
            for stage in case["stages"]
        )
        print(f"{case['true_max']},{stages},{case['final']['low_dimensional']['prediction']}")
    print(
        "output_top3_explained_variance,"
        f"{data['basis']['top3_cumulative_explained_variance']:.12f}"
    )
    print(
        "principal_angles_degrees,"
        + ",".join(
            f"{angle:.6f}"
            for angle in data["basis"][
                "principal_angles_to_digit_unembedding_top3_subspace_degrees"
            ]
        )
    )
    print(f"display_scale,{data['display_scale_for_head_writes']:.9f}")
    print(f"wrote,{JSON_OUT}")
    print(f"wrote,{HTML_OUT}")


if __name__ == "__main__":
    main()
