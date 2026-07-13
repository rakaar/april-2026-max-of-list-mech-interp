#!/usr/bin/env python3
"""Reproduce the quantitative checks used by the main-results supplement."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from model1_output_pca_readout_accuracy import (
    BATCH_SIZE,
    KS,
    extract_ans_head_values,
    load_model,
    tokenize,
)
from model1_unembedding_pca_readout_accuracy import (
    captured_fractions,
    evaluate,
    fit_centered_pca,
    per_head_output_matrices,
)


ROOT = Path(__file__).resolve().parents[2]
JSON_OUT = ROOT / "docs" / "assets" / "model1_main_results_supplementary.json"
PNG_OUT = ROOT / "docs" / "assets" / "model1_main_results_pc_alignment.png"


def aligned_pc_comparison(
    unembedding_basis: torch.Tensor, output_basis: torch.Tensor
) -> dict:
    q_u = unembedding_basis[:, :3].clone()
    q_o = output_basis[:, :3].clone()
    signs = []
    for pc_idx in range(3):
        sign = 1.0 if float(q_u[:, pc_idx] @ q_o[:, pc_idx]) >= 0.0 else -1.0
        q_o[:, pc_idx] *= sign
        signs.append(int(sign))

    cosine_matrix = q_u.T @ q_o
    overlap_singular_values = torch.linalg.svdvals(cosine_matrix)
    principal_angles = torch.rad2deg(
        torch.arccos(overlap_singular_values.clamp(-1.0, 1.0))
    )
    return {
        "output_pc_signs_applied": signs,
        "pc_cosine_matrix": cosine_matrix,
        "same_index_pc_cosines": torch.diagonal(cosine_matrix),
        "principal_angles_degrees": principal_angles,
        "aligned_output_basis": q_o,
    }


def canonical_angle_examples(
    model,
    matrices: torch.Tensor,
    basis: torch.Tensor,
) -> list[dict]:
    """Measure the max-1/max-2 geometry shown by the direct-head interactive."""

    layer = model.layers[0]
    positions = torch.arange(11).unsqueeze(0)
    causal_mask = torch.tril(torch.ones(11, 11)).unsqueeze(0)
    unembedding = model.unembed.weight.detach().cpu().double()
    centered_unembedding = unembedding - unembedding.mean(dim=0, keepdim=True)
    unembedding_3d = centered_unembedding @ basis
    rows = []

    for target in (1, 2):
        numbers = torch.tensor([[0, 0, target, 0, 0]])
        tokens = tokenize(numbers)
        with torch.no_grad():
            residual = model.tok_embed(tokens) + model.pos_embed(positions)
            source_values = torch.stack(
                [
                    (residual @ head.W_V.weight.detach().T)[0].double()
                    for head in layer.heads
                ]
            )
            source_writes = torch.einsum("hpd,hdm->hpm", source_values, matrices)
            chosen = [source_writes[head_idx, 10] for head_idx in range(4)]

            if target == 1:
                h3_values, _ = layer.heads[3](residual, causal_mask)
                chosen[3] = h3_values[0, 10].double() @ matrices[3]
                routing = "H0-H2 -> [ANS]; H3 uses its measured soft [ANS]/1 row"
            else:
                chosen[3] = source_writes[3, 5]
                routing = "H0-H2 -> [ANS]; H3 -> digit 2 (one-hot)"

        head_sum = torch.stack(chosen).sum(dim=0)
        final_state = residual[0, -1].double() + head_sum
        final_3d = final_state @ basis

        for candidate in (target, target + 1):
            candidate_vector = unembedding_3d[candidate]
            dot_product = float(final_3d @ candidate_vector)
            cosine = float(
                F.cosine_similarity(final_3d, candidate_vector, dim=0)
            )
            rows.append(
                {
                    "true_max": target,
                    "numbers": numbers[0].tolist(),
                    "routing": routing,
                    "candidate_digit": candidate,
                    "cosine": cosine,
                    "projected_unembedding_norm": float(candidate_vector.norm()),
                    "dot_product": dot_product,
                    "is_dot_product_winner_of_pair": candidate == target,
                    "is_cosine_winner_of_pair": candidate == target + 1,
                }
            )

    return rows


def exhaustive_state_and_h1_checks(
    model, matrices: torch.Tensor
) -> dict:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    all_numbers = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)])
    all_labels = all_numbers.max(dim=1).values
    total = len(all_numbers)
    unembedding = model.unembed.weight.detach()
    matrices_device = matrices.to(device=device, dtype=dtype)

    correct = {
        "head_sum_only": 0,
        "full_model": 0,
        "h1_ablated_full_model": 0,
        "h1_ablated_head_sum_only": 0,
    }
    h1_failures = []

    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            numbers = all_numbers[start:end].to(device)
            labels = all_labels[start:end].to(device)
            tokens = tokenize(numbers)
            positions = torch.arange(tokens.shape[1], device=device).unsqueeze(0)
            residual = model.tok_embed(tokens) + model.pos_embed(positions)

            head_values = extract_ans_head_values(model, tokens)
            head_writes = torch.einsum(
                "bhd,hdr->bhr", head_values, matrices_device
            )
            head_sum = head_writes.sum(dim=1)
            h1_ablated_sum = head_sum - head_writes[:, 1]
            ans_residual = residual[:, -1]

            states = {
                "head_sum_only": head_sum,
                "full_model": ans_residual + head_sum,
                "h1_ablated_full_model": ans_residual + h1_ablated_sum,
                "h1_ablated_head_sum_only": h1_ablated_sum,
            }
            predictions = {}
            logits_by_variant = {}
            for name, state in states.items():
                logits = state @ unembedding.T
                prediction = logits.argmax(dim=1)
                logits_by_variant[name] = logits
                predictions[name] = prediction
                correct[name] += int((prediction == labels).sum())

            failed = predictions["h1_ablated_full_model"] != labels
            for local_idx in failed.nonzero(as_tuple=False).flatten().tolist():
                logits = logits_by_variant["h1_ablated_full_model"][local_idx]
                top2 = torch.topk(logits, 2)
                h1_failures.append(
                    {
                        "numbers": numbers[local_idx].cpu().tolist(),
                        "true_max": int(labels[local_idx]),
                        "prediction": int(
                            predictions["h1_ablated_full_model"][local_idx]
                        ),
                        "top_logit": float(top2.values[0]),
                        "runner_up_token": int(top2.indices[1]),
                        "runner_up_logit": float(top2.values[1]),
                        "prediction_margin": float(top2.values[0] - top2.values[1]),
                        "original_prediction": int(predictions["full_model"][local_idx]),
                    }
                )

    if correct["head_sum_only"] != total:
        raise AssertionError(
            f"head-sum-only readout failed: {correct['head_sum_only']}/{total}"
        )
    if correct["full_model"] != total:
        raise AssertionError(f"full model failed: {correct['full_model']}/{total}")
    if correct["h1_ablated_full_model"] != 99_997:
        raise AssertionError(
            "unexpected H1-ablation result: "
            f"{correct['h1_ablated_full_model']}/{total}"
        )

    return {
        "n_inputs": total,
        "candidate_vocabulary_size": int(unembedding.shape[0]),
        "variants": {
            name: {
                "correct": value,
                "total": total,
                "accuracy": value / total,
            }
            for name, value in correct.items()
        },
        "h1_ablated_full_model_failures": h1_failures,
    }


def max8_h0_attention_interventions(
    model,
    matrices: torch.Tensor,
    output_basis: torch.Tensor,
    unembedding_basis: torch.Tensor,
) -> dict:
    """Test whether H0's partial max-8 read matters in 64D or either 3D basis."""

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    all_numbers = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)])
    all_labels = all_numbers.max(dim=1).values
    numbers = all_numbers[all_labels == 8]
    total = len(numbers)
    if total != 26_281:
        raise AssertionError(f"unexpected number of max-8 inputs: {total}")

    matrices_device = matrices.to(device=device, dtype=dtype)
    unembedding = model.unembed.weight.detach()
    centered_unembedding = unembedding - unembedding.mean(dim=0, keepdim=True)
    bases = {
        "output_pca_3d_head_sum_only": output_basis[:, :3].to(
            device=device, dtype=dtype
        ),
        "unembedding_pca_3d_head_sum_only": unembedding_basis[:, :3].to(
            device=device, dtype=dtype
        ),
    }
    reduced_outputs = {
        name: matrices_device @ basis for name, basis in bases.items()
    }
    reduced_unembeddings = {
        name: centered_unembedding @ basis for name, basis in bases.items()
    }
    readout_names = ["full_64d_with_ans_residual", *bases]
    intervention_names = (
        "actual_attention",
        "move_h0_8_mass_to_ans",
        "force_h0_onehot_ans",
    )
    stats = {
        readout: {
            intervention: {
                "correct": 0,
                "prediction_counts": torch.zeros(
                    len(unembedding), dtype=torch.long
                ),
                "target_margin_sum": 0.0,
                "minimum_target_margin": float("inf"),
            }
            for intervention in intervention_names
        }
        for readout in readout_names
    }
    h0_attention_mass_sums = {"8": 0.0, "7": 0.0, "ANS": 0.0}

    def update_stats(row: dict, logits: torch.Tensor) -> None:
        predictions = logits.argmax(dim=1)
        row["correct"] += int((predictions == 8).sum())
        row["prediction_counts"] += torch.bincount(
            predictions.cpu(), minlength=len(unembedding)
        )
        target_logits = logits[:, 8]
        competitors = logits.clone()
        competitors[:, 8] = -torch.inf
        margins = target_logits - competitors.max(dim=1).values
        row["target_margin_sum"] += float(margins.sum())
        row["minimum_target_margin"] = min(
            row["minimum_target_margin"], float(margins.min())
        )

    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_numbers = numbers[start:end].to(device)
            tokens = tokenize(batch_numbers)
            positions = torch.arange(tokens.shape[1], device=device).unsqueeze(0)
            residual = model.tok_embed(tokens) + model.pos_embed(positions)
            ans_residual = residual[:, -1]
            causal_mask = torch.tril(
                torch.ones(tokens.shape[1], tokens.shape[1], device=device)
            ).unsqueeze(0)

            actual_values = []
            h0_move_8_to_ans = None
            h0_onehot_ans = None
            for head_index, head in enumerate(model.layers[0].heads):
                output, attention = head(residual, causal_mask)
                source_values = residual @ head.W_V.weight.detach().T
                actual_values.append(output[:, -1])
                if head_index != 0:
                    continue

                actual_h0_row = attention[:, -1]
                is_eight = tokens == 8
                h0_attention_mass_sums["8"] += float(
                    (actual_h0_row * is_eight).sum()
                )
                h0_attention_mass_sums["7"] += float(
                    (actual_h0_row * (tokens == 7)).sum()
                )
                h0_attention_mass_sums["ANS"] += float(actual_h0_row[:, -1].sum())

                modified_row = actual_h0_row.clone()
                removed_mass = (modified_row * is_eight).sum(dim=1)
                modified_row.masked_fill_(is_eight, 0.0)
                modified_row[:, -1] += removed_mass
                if not torch.allclose(
                    modified_row.sum(dim=1),
                    torch.ones_like(removed_mass),
                    atol=1e-6,
                    rtol=0.0,
                ):
                    raise AssertionError("modified H0 attention row is not normalized")
                h0_move_8_to_ans = torch.einsum(
                    "bn,bnd->bd", modified_row, source_values
                )
                h0_onehot_ans = source_values[:, -1]

            if h0_move_8_to_ans is None or h0_onehot_ans is None:
                raise AssertionError("failed to construct H0 interventions")
            variants = {
                "actual_attention": actual_values,
                "move_h0_8_mass_to_ans": [
                    h0_move_8_to_ans,
                    *actual_values[1:],
                ],
                "force_h0_onehot_ans": [h0_onehot_ans, *actual_values[1:]],
            }

            for intervention, values in variants.items():
                head_values = torch.stack(values, dim=1)
                head_writes = torch.einsum(
                    "bhd,hdr->bhr", head_values, matrices_device
                )
                head_sum = head_writes.sum(dim=1)
                full_logits = (ans_residual + head_sum) @ unembedding.T
                update_stats(
                    stats["full_64d_with_ans_residual"][intervention],
                    full_logits,
                )

                for readout_name, basis in bases.items():
                    low_sum = torch.einsum(
                        "bhd,hdk->bk",
                        head_values,
                        reduced_outputs[readout_name],
                    )
                    reference = head_sum @ basis
                    if not torch.allclose(
                        low_sum, reference, rtol=1e-5, atol=2e-4
                    ):
                        error = float((low_sum - reference).abs().max())
                        raise AssertionError(
                            f"reduced output route disagrees for {readout_name}: {error}"
                        )
                    low_logits = (
                        low_sum @ reduced_unembeddings[readout_name].T
                    )
                    update_stats(stats[readout_name][intervention], low_logits)

    serialized_readouts = {}
    for readout_name, interventions in stats.items():
        serialized_readouts[readout_name] = {}
        for intervention, row in interventions.items():
            if row["correct"] != total:
                raise AssertionError(
                    f"max-8 intervention failed for {readout_name}/{intervention}: "
                    f"{row['correct']}/{total}"
                )
            serialized_readouts[readout_name][intervention] = {
                "correct": row["correct"],
                "total": total,
                "accuracy": row["correct"] / total,
                "prediction_distribution": {
                    str(token): int(count)
                    for token, count in enumerate(row["prediction_counts"])
                    if int(count) > 0
                },
                "mean_target_margin": row["target_margin_sum"] / total,
                "minimum_target_margin": row["minimum_target_margin"],
            }

    return {
        "scope": "all five-digit inputs with true maximum 8",
        "n_inputs": total,
        "h0_mean_attention_mass": {
            token: value / total
            for token, value in h0_attention_mass_sums.items()
        },
        "interventions": {
            "move_h0_8_mass_to_ans": (
                "For each input, set H0 attention to every source token 8 to zero "
                "and add exactly that removed probability to the ANS self position."
            ),
            "force_h0_onehot_ans": (
                "Replace the complete H0 ANS-query attention row by one-hot ANS self."
            ),
            "other_heads": "Keep the measured H1, H2, and H3 attention outputs unchanged.",
        },
        "three_dimensional_readout": {
            "output_pca_basis": (
                "Top three PCs fitted to centered rows of the stacked 64x64 output matrix"
            ),
            "unembedding_pca_basis": (
                "Top three PCs fitted to centered rows of the full 14x64 unembedding"
            ),
            "residual_policy": (
                "Use only the projected sum of head outputs, matching the main "
                "low-dimensional computation; do not add the initial ANS residual."
            ),
            "candidate_vocabulary_size": int(len(unembedding)),
        },
        "readouts": serialized_readouts,
    }


def plot_alignment(comparison: dict) -> None:
    cosine_matrix = comparison["pc_cosine_matrix"].numpy()

    fig, ax = plt.subplots(figsize=(6.9, 5.8), constrained_layout=True)
    image = ax.imshow(cosine_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    for row in range(3):
        for column in range(3):
            value = cosine_matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.58 else "#172126",
                fontsize=12,
                fontweight="bold" if row == column else "normal",
            )

    ax.set_xticks(range(3), ["$W_O$ PC1", "$W_O$ PC2", "$W_O$ PC3"])
    ax.set_yticks(
        range(3),
        ["$W_U$ PC1", "$W_U$ PC2", "$W_U$ PC3"],
    )
    ax.set_title("Cosine similarity of full-vocabulary $W_U$ and $W_O$ PCs", pad=14)
    ax.set_xlabel("Output-matrix principal directions")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.05)
    colorbar.set_label("64D cosine similarity")
    fig.savefig(PNG_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def tensor_values(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def main() -> None:
    torch.manual_seed(0)
    model, config = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    matrices = per_head_output_matrices(model)
    output_matrix = matrices.flatten(0, 1)
    unembedding = model.unembed.weight.detach().cpu().double()
    unembedding_pca = fit_centered_pca(unembedding)
    output_pca = fit_centered_pca(output_matrix)
    q_u = unembedding_pca["basis"]
    q_o = output_pca["basis"]

    unembedding_capture = {
        str(k): float(unembedding_pca["explained"][:k].sum()) for k in KS
    }
    output_capture = captured_fractions(output_pca["centered"], q_u)
    reverse_evaluation = evaluate(model, matrices, q_u)
    comparison = aligned_pc_comparison(q_u, q_o)
    angle_rows = canonical_angle_examples(
        model.cpu(), matrices, comparison["aligned_output_basis"]
    )
    model = model.to(device)
    exhaustive_checks = exhaustive_state_and_h1_checks(model, matrices)
    max8_h0_checks = max8_h0_attention_interventions(
        model,
        matrices,
        q_o,
        q_u,
    )

    result = {
        "description": (
            "Reproducible checks for the supplementary section of the main-results "
            "page: full-vocabulary unembedding-PCA readout, PC alignment, canonical "
            "max-1/max-2 angle examples, head-sum-only readout, H1 ablation, "
            "and max-8 H0 attention interventions in 64D and both 3D bases."
        ),
        "hf_repo": "andyrdt/04_2026_puzzle_1a",
        "model_config": config,
        "device": str(device),
        "unembedding_pca_readout": {
            "basis_source": "centered rows of the full 14x64 unembedding weight",
            "unembedding_cumulative_variance": unembedding_capture,
            "output_matrix_cumulative_variance_in_this_basis": output_capture,
            "by_k": reverse_evaluation["by_k"],
        },
        "pc_alignment": {
            "definition": (
                "Rows are full-vocabulary unembedding PCs; columns are stacked-output "
                "PCs. Output-PC signs are chosen for a positive same-index cosine."
            ),
            "output_pc_signs_applied": comparison["output_pc_signs_applied"],
            "pc_cosine_matrix": tensor_values(comparison["pc_cosine_matrix"]),
            "same_index_pc_cosines": tensor_values(
                comparison["same_index_pc_cosines"]
            ),
            "principal_angles_degrees": tensor_values(
                comparison["principal_angles_degrees"]
            ),
        },
        "canonical_output_pca_angle_examples": {
            "basis": "top three centered stacked-output PCs",
            "unembedding_centering": (
                "subtract the mean of all 14 vocabulary rows before projection"
            ),
            "state": "R_final[-1,:] = R[-1,:] + sum_h z_h",
            "rows": angle_rows,
        },
        "exhaustive_state_and_h1_checks": exhaustive_checks,
        "max8_h0_attention_interventions": max8_h0_checks,
    }
    JSON_OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot_alignment(comparison)

    print("k,unembedding_variance,output_variance,full_vocab_accuracy")
    for k in KS:
        print(
            f"{k},{unembedding_capture[str(k)]:.9f},{output_capture[str(k)]:.9f},"
            f"{reverse_evaluation['by_k'][str(k)]['accuracy']:.9f}"
        )
    print("angle_rows")
    for row in angle_rows:
        print(
            f"max={row['true_max']},candidate={row['candidate_digit']},"
            f"cosine={row['cosine']:.9f},norm={row['projected_unembedding_norm']:.9f},"
            f"dot={row['dot_product']:.9f}"
        )
    for name, values in exhaustive_checks["variants"].items():
        print(f"{name},{values['correct']}/{values['total']},{values['accuracy']:.9f}")
    print("h1_failures", exhaustive_checks["h1_ablated_full_model_failures"])
    print("max8_h0_mean_attention", max8_h0_checks["h0_mean_attention_mass"])
    print("max8_h0_interventions")
    for readout_name, interventions in max8_h0_checks["readouts"].items():
        for intervention, values in interventions.items():
            print(
                f"{readout_name},{intervention},"
                f"{values['correct']}/{values['total']},{values['accuracy']:.9f},"
                f"min_margin={values['minimum_target_margin']:.9f}"
            )
    print(f"wrote,{JSON_OUT}")
    print(f"wrote,{PNG_OUT}")


if __name__ == "__main__":
    main()
