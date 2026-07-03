#!/usr/bin/env python3
"""Project the piecewise attention scheme into the head-sum PCA plane."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_piecewise_scheme_pca_projection.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_piecewise_scheme_pca_projection.json"
NUMBER_POSITIONS = torch.tensor([1, 3, 5, 7, 9])
BATCH_SIZE = 4096


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
    return model


def make_inputs(device: str):
    nums = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)]).to(device)
    labels = nums.max(dim=1).values
    tokens = torch.empty((nums.shape[0], 11), dtype=torch.long, device=device)
    tokens[:, 0] = 10
    tokens[:, 1] = nums[:, 0]
    tokens[:, 2] = 11
    tokens[:, 3] = nums[:, 1]
    tokens[:, 4] = 11
    tokens[:, 5] = nums[:, 2]
    tokens[:, 6] = 11
    tokens[:, 7] = nums[:, 3]
    tokens[:, 8] = 11
    tokens[:, 9] = nums[:, 4]
    tokens[:, 10] = 12
    return nums, tokens, labels


def digit_logits(vec: torch.Tensor, model) -> torch.Tensor:
    return vec @ model.unembed.weight.detach()[:10].T


def apply_w_o_slices(layer, head_values: list[torch.Tensor]) -> torch.Tensor:
    w_o = layer.W_O.weight.detach()
    d_head = head_values[0].shape[-1]
    pieces = []
    for head_idx, value in enumerate(head_values):
        w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
        pieces.append(value @ w_o_head.T)
    return sum(pieces)


def collect_actual_and_scheme_head_sums(model):
    device = next(model.parameters()).device
    nums, tokens, labels = make_inputs(str(device))
    total = tokens.shape[0]
    number_positions = NUMBER_POSITIONS.to(device)

    all_actual = []
    all_scheme = []

    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            x = tokens[start:end]
            n = nums[start:end]
            y = labels[start:end]
            batch = end - start
            batch_idx = torch.arange(batch, device=device)
            is_max_slot = n == y[:, None]

            seq_len = x.shape[1]
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            resid = model.tok_embed(x) + model.pos_embed(positions)
            layer = model.layers[0]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)

            self_values = []
            max_values = []
            actual_values = []

            for head in layer.heads:
                out, attn = head(resid, mask)
                attn_row = attn[:, 10, :]
                source_values = resid @ head.W_V.weight.detach().T
                number_attn = attn_row[:, number_positions]
                max_attn = number_attn.masked_fill(~is_max_slot, -1.0)
                max_slot = max_attn.argmax(dim=1)
                max_pos = number_positions[max_slot]

                self_values.append(source_values[:, 10, :])
                max_values.append(source_values[batch_idx, max_pos])
                actual_values.append(out[:, 10, :])

            h0_choice = torch.where((y == 9).unsqueeze(1), max_values[0], self_values[0])
            h1_choice = self_values[1]
            h2_choice = torch.where((y >= 7).unsqueeze(1), max_values[2], self_values[2])
            h3_not_one = torch.where((y == 0).unsqueeze(1), self_values[3], max_values[3])
            h3_choice = torch.where((y == 1).unsqueeze(1), actual_values[3], h3_not_one)

            actual_head_sum = apply_w_o_slices(layer, actual_values)
            scheme_head_sum = apply_w_o_slices(
                layer,
                [h0_choice, h1_choice, h2_choice, h3_choice],
            )

            all_actual.append(actual_head_sum.cpu())
            all_scheme.append(scheme_head_sum.cpu())

    return labels.cpu(), torch.cat(all_actual), torch.cat(all_scheme)


def pca_fit(x: torch.Tensor):
    mean = x.mean(dim=0)
    centered = x - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    explained = energy / energy.sum()
    scores = centered @ vh.T
    return mean, vh, singular_values, explained, scores


def project_to_basis(x: torch.Tensor, mean: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    return (x - mean) @ directions.T


def reconstruct_from_scores(scores: torch.Tensor, mean: torch.Tensor, directions: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return mean.unsqueeze(0).expand(scores.shape[0], -1)
    return mean + scores[:, :k] @ directions[:k]


def summarize_by_max(labels: torch.Tensor, actual_scores: torch.Tensor, scheme_scores: torch.Tensor):
    out = {}
    for max_value in range(10):
        mask = labels == max_value
        actual_xy = actual_scores[mask, :2]
        scheme_xy = scheme_scores[mask, :2]
        actual_mean = actual_xy.mean(dim=0)
        scheme_mean = scheme_xy.mean(dim=0)
        delta = scheme_mean - actual_mean
        out[str(max_value)] = {
            "count": int(mask.sum()),
            "actual_pc1_mean": float(actual_mean[0]),
            "actual_pc2_mean": float(actual_mean[1]),
            "actual_pc1_std": float(actual_xy[:, 0].std(unbiased=True)) if int(mask.sum()) > 1 else 0.0,
            "actual_pc2_std": float(actual_xy[:, 1].std(unbiased=True)) if int(mask.sum()) > 1 else 0.0,
            "scheme_pc1_mean": float(scheme_mean[0]),
            "scheme_pc2_mean": float(scheme_mean[1]),
            "scheme_pc1_std": float(scheme_xy[:, 0].std(unbiased=True)) if int(mask.sum()) > 1 else 0.0,
            "scheme_pc2_std": float(scheme_xy[:, 1].std(unbiased=True)) if int(mask.sum()) > 1 else 0.0,
            "scheme_minus_actual_pc1": float(delta[0]),
            "scheme_minus_actual_pc2": float(delta[1]),
            "scheme_actual_distance_2d": float(torch.linalg.vector_norm(delta)),
        }
    return out


def accuracy_from_vectors(vecs: torch.Tensor, labels: torch.Tensor, model) -> float:
    logits = digit_logits(vecs, model.cpu())
    pred = logits.argmax(dim=1)
    return float((pred == labels).float().mean())


def make_decision_grid(mean: torch.Tensor, directions: torch.Tensor, model, xlim, ylim):
    xs = np.linspace(xlim[0], xlim[1], 360)
    ys = np.linspace(ylim[0], ylim[1], 320)
    xx, yy = np.meshgrid(xs, ys)
    grid_scores = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1),
        dtype=mean.dtype,
    )
    grid_vecs = mean + grid_scores @ directions[:2]
    logits = digit_logits(grid_vecs, model.cpu())
    pred = logits.argmax(dim=1).numpy().reshape(xx.shape)
    return xs, ys, pred


def recipe_label(max_value: int) -> str:
    if max_value == 0:
        return "all [ANS]"
    if max_value == 1:
        return "H3 soft 1 + [ANS]"
    if 2 <= max_value <= 6:
        return "H3 max"
    if 7 <= max_value <= 8:
        return "H2+H3 max"
    return "H0+H2+H3 max"


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device)
    labels, actual_head_sum, scheme_head_sum = collect_actual_and_scheme_head_sums(model)
    model = model.cpu()

    mean, directions, singular_values, explained, actual_scores = pca_fit(actual_head_sum)
    scheme_scores = project_to_basis(scheme_head_sum, mean, directions)

    actual_logits = digit_logits(actual_head_sum, model)
    scheme_logits = digit_logits(scheme_head_sum, model)
    actual_pred = actual_logits.argmax(dim=1)
    scheme_pred = scheme_logits.argmax(dim=1)

    actual_top2_recon = reconstruct_from_scores(actual_scores, mean, directions, 2)
    scheme_top2_recon = reconstruct_from_scores(scheme_scores, mean, directions, 2)
    actual_top2_pred = digit_logits(actual_top2_recon, model).argmax(dim=1)
    scheme_top2_pred = digit_logits(scheme_top2_recon, model).argmax(dim=1)

    by_max = summarize_by_max(labels, actual_scores, scheme_scores)
    actual_centers = torch.tensor(
        [[by_max[str(i)]["actual_pc1_mean"], by_max[str(i)]["actual_pc2_mean"]] for i in range(10)]
    )
    scheme_centers = torch.tensor(
        [[by_max[str(i)]["scheme_pc1_mean"], by_max[str(i)]["scheme_pc2_mean"]] for i in range(10)]
    )

    result = {
        "description": (
            "PCA basis is fit on actual 100000 head-sum vectors at [ANS] after W_O. "
            "Piecewise scheme vectors use H0/H1/H2/H3 source choices from the complete "
            "attention abstraction, with H3 actual soft value for true max 1. Both actual "
            "and scheme vectors are projected into the same actual PC1/PC2 basis."
        ),
        "n_inputs": int(labels.shape[0]),
        "head_sum_shape": list(actual_head_sum.shape),
        "pca_explained_variance": [float(v) for v in explained],
        "pca_cumulative_explained_variance": [float(v) for v in torch.cumsum(explained, dim=0)],
        "accuracy": {
            "actual_head_sum_only": float((actual_pred == labels).float().mean()),
            "scheme_head_sum_only": float((scheme_pred == labels).float().mean()),
            "actual_head_sum_top2_pc_reconstruction": float((actual_top2_pred == labels).float().mean()),
            "scheme_head_sum_top2_pc_reconstruction": float((scheme_top2_pred == labels).float().mean()),
        },
        "prediction_distribution": {
            "scheme_head_sum_only": {
                str(d): int((scheme_pred == d).sum()) for d in range(10) if int((scheme_pred == d).sum()) > 0
            },
            "scheme_head_sum_top2_pc_reconstruction": {
                str(d): int((scheme_top2_pred == d).sum())
                for d in range(10)
                if int((scheme_top2_pred == d).sum()) > 0
            },
        },
        "centers_by_true_max": by_max,
        "attention_recipe": {str(i): recipe_label(i) for i in range(10)},
    }
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    x_all = torch.cat([actual_scores[:, 0], scheme_scores[:, 0], actual_centers[:, 0], scheme_centers[:, 0]])
    y_all = torch.cat([actual_scores[:, 1], scheme_scores[:, 1], actual_centers[:, 1], scheme_centers[:, 1]])
    x_pad = 0.08 * float(x_all.max() - x_all.min())
    y_pad = 0.08 * float(y_all.max() - y_all.min())
    xlim = (float(x_all.min() - x_pad), float(x_all.max() + x_pad))
    ylim = (float(y_all.min() - y_pad), float(y_all.max() + y_pad))
    xs, ys, decision = make_decision_grid(mean, directions, model, xlim, ylim)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    for ax in axes:
        ax.imshow(
            decision,
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            origin="lower",
            cmap=cmap,
            alpha=0.16,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("actual head-sum PC1 score")
        ax.set_ylabel("actual head-sum PC2 score")

    sample_indices = []
    generator = torch.Generator().manual_seed(0)
    for max_value in range(10):
        idx = (labels == max_value).nonzero(as_tuple=False).flatten()
        take = min(len(idx), 1000)
        sample_indices.append(idx[torch.randperm(len(idx), generator=generator)[:take]])
    sample = torch.cat(sample_indices)
    axes[0].scatter(
        actual_scores[sample, 0].numpy(),
        actual_scores[sample, 1].numpy(),
        c=labels[sample].numpy(),
        cmap=cmap,
        s=8,
        alpha=0.42,
        linewidths=0,
    )
    axes[0].plot(actual_centers[:, 0].numpy(), actual_centers[:, 1].numpy(), color="black", linewidth=1.1)
    for digit, (x, y) in enumerate(actual_centers.tolist()):
        axes[0].text(x, y, str(digit), ha="center", va="center", fontsize=11, weight="bold")
    axes[0].set_title("Actual head-sum vectors in the 2d answer plane")

    recipe_colors = {
        "all [ANS]": "#4b5563",
        "H3 soft 1 + [ANS]": "#7c3aed",
        "H3 max": "#2563eb",
        "H2+H3 max": "#059669",
        "H0+H2+H3 max": "#dc2626",
    }
    seen_labels: set[str] = set()
    for digit in range(10):
        actual_x, actual_y = actual_centers[digit].tolist()
        scheme_x, scheme_y = scheme_centers[digit].tolist()
        label = recipe_label(digit)
        color = recipe_colors[label]
        axes[1].plot(actual_x, actual_y, marker="o", markersize=7, color=color, alpha=0.55)
        axes[1].plot(
            scheme_x,
            scheme_y,
            marker="x",
            markersize=9,
            markeredgewidth=2.2,
            color=color,
            label=label if label not in seen_labels else None,
        )
        axes[1].annotate(
            "",
            xy=(scheme_x, scheme_y),
            xytext=(actual_x, actual_y),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1, "alpha": 0.65},
        )
        axes[1].text(scheme_x, scheme_y, f" {digit}", ha="left", va="center", fontsize=10, weight="bold")
        seen_labels.add(label)
    axes[1].plot(scheme_centers[:, 0].numpy(), scheme_centers[:, 1].numpy(), color="black", linewidth=1.0, alpha=0.55)
    axes[1].set_title("Piecewise attention recipes projected onto same plane")
    axes[1].legend(loc="best", fontsize=9, frameon=True)

    fig.suptitle(
        "Model 1: where each max-number recipe lands in the 2d head-sum decision plane",
        fontsize=14,
    )
    fig.savefig(OUT, dpi=180)

    print("accuracy")
    for key, value in result["accuracy"].items():
        print(f"{key},{value:.6f}")
    print("centers")
    print("max,actual_pc1,actual_pc2,scheme_pc1,scheme_pc2,recipe")
    for digit in range(10):
        row = by_max[str(digit)]
        print(
            f"{digit},"
            f"{row['actual_pc1_mean']:.6f},"
            f"{row['actual_pc2_mean']:.6f},"
            f"{row['scheme_pc1_mean']:.6f},"
            f"{row['scheme_pc2_mean']:.6f},"
            f"{recipe_label(digit)}"
        )
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
