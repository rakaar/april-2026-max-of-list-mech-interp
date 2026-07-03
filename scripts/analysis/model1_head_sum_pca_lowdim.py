#!/usr/bin/env python3
"""Test whether Model 1 head-sum outputs are effectively low-dimensional."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_head_sum_pca_lowdim.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_head_sum_pca_lowdim.json"


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


def tokenize(nums: list[int]) -> list[int]:
    return [10, nums[0], 11, nums[1], 11, nums[2], 11, nums[3], 11, nums[4], 12]


def number_logits(vec: torch.Tensor, model) -> torch.Tensor:
    return vec @ model.unembed.weight.detach()[:10].T


def collect_head_sums(model):
    all_nums = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)])
    head_sums = []
    labels = []
    chunk_size = 4096

    with torch.no_grad():
        for start in range(0, len(all_nums), chunk_size):
            nums_t = all_nums[start : start + chunk_size]
            chunk_labels = nums_t.max(dim=1).values
            tokens = torch.tensor([tokenize(row.tolist()) for row in nums_t], dtype=torch.long)
            batch, seq_len = tokens.shape
            positions = torch.arange(seq_len).unsqueeze(0)
            resid = model.tok_embed(tokens) + model.pos_embed(positions)
            layer = model.layers[0]
            w_o = layer.W_O.weight.detach()
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

            head_vectors = []
            for head_idx, head in enumerate(layer.heads):
                head_values, _ = head(resid, mask)
                d_head = head.d_head
                w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
                head_vectors.append(head_values[:, 10, :] @ w_o_head.T)

            head_sums.append(sum(head_vectors).cpu())
            labels.append(chunk_labels.cpu())

    return all_nums, torch.cat(labels), torch.cat(head_sums)


def pca_fit(x: torch.Tensor):
    mean = x.mean(dim=0)
    centered = x - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.square()
    explained = energy / energy.sum()
    scores = centered @ vh.T
    return {
        "mean": mean,
        "centered": centered,
        "singular_values": singular_values,
        "directions": vh,
        "explained": explained,
        "scores": scores,
    }


def projection_accuracy(centered: torch.Tensor, mean: torch.Tensor, directions: torch.Tensor, labels: torch.Tensor, model, ks: list[int]):
    out = {}
    w_u = model.unembed.weight.detach()[:10]
    for k in ks:
        if k == 0:
            recon = mean.unsqueeze(0).expand_as(centered)
        else:
            basis = directions[:k]
            recon = mean + (centered @ basis.T) @ basis
        logits = recon @ w_u.T
        pred = logits.argmax(dim=1)
        out[str(k)] = {
            "accuracy": float((pred == labels).float().mean()),
            "prediction_distribution": {
                str(d): int((pred == d).sum()) for d in range(10) if int((pred == d).sum()) > 0
            },
        }
    return out


def logit_projection_accuracy(logit_centered: torch.Tensor, logit_mean: torch.Tensor, directions: torch.Tensor, labels: torch.Tensor, ks: list[int]):
    out = {}
    for k in ks:
        if k == 0:
            recon = logit_mean.unsqueeze(0).expand_as(logit_centered)
        else:
            basis = directions[:k]
            recon = logit_mean + (logit_centered @ basis.T) @ basis
        pred = recon.argmax(dim=1)
        out[str(k)] = {
            "accuracy": float((pred == labels).float().mean()),
            "prediction_distribution": {
                str(d): int((pred == d).sum()) for d in range(10) if int((pred == d).sum()) > 0
            },
        }
    return out


def min_k_for_variance(explained: torch.Tensor, threshold: float) -> int:
    return int((torch.cumsum(explained, dim=0) >= threshold).nonzero()[0].item() + 1)


def main() -> None:
    torch.manual_seed(0)
    model = load_model()
    _, labels, head_sum = collect_head_sums(model)
    head_logits = number_logits(head_sum, model)
    full_head_acc = float((head_logits.argmax(dim=1) == labels).float().mean())

    head_pca = pca_fit(head_sum)
    logit_pca = pca_fit(head_logits)

    head_ks = list(range(0, 21)) + [24, 32, 48, 64]
    logit_ks = list(range(0, 11))
    head_projection = projection_accuracy(
        head_pca["centered"],
        head_pca["mean"],
        head_pca["directions"],
        labels,
        model,
        head_ks,
    )
    logit_projection = logit_projection_accuracy(
        logit_pca["centered"],
        logit_pca["mean"],
        logit_pca["directions"],
        labels,
        logit_ks,
    )

    by_true_max = {}
    for max_value in range(10):
        mask = labels == max_value
        by_true_max[str(max_value)] = {
            "count": int(mask.sum()),
            "pc1_mean": float(head_pca["scores"][mask, 0].mean()),
            "pc2_mean": float(head_pca["scores"][mask, 1].mean()),
            "pc1_std": float(head_pca["scores"][mask, 0].std(unbiased=True)) if int(mask.sum()) > 1 else 0.0,
            "pc2_std": float(head_pca["scores"][mask, 1].std(unbiased=True)) if int(mask.sum()) > 1 else 0.0,
        }

    result = {
        "description": (
            "PCA/SVD on all 100000 [ANS] head-sum vectors after W_O. "
            "Projection accuracies reconstruct either the 64d head-sum vector before unembedding "
            "or the 10d head-sum logits directly."
        ),
        "head_sum_shape": list(head_sum.shape),
        "head_sum_only_accuracy": full_head_acc,
        "head_sum_pca": {
            "singular_values": [float(v) for v in head_pca["singular_values"]],
            "explained_variance": [float(v) for v in head_pca["explained"]],
            "cumulative_explained_variance": [float(v) for v in torch.cumsum(head_pca["explained"], dim=0)],
            "k_for_90pct": min_k_for_variance(head_pca["explained"], 0.90),
            "k_for_95pct": min_k_for_variance(head_pca["explained"], 0.95),
            "k_for_99pct": min_k_for_variance(head_pca["explained"], 0.99),
        },
        "head_sum_projection_accuracy": head_projection,
        "logit_pca": {
            "singular_values": [float(v) for v in logit_pca["singular_values"]],
            "explained_variance": [float(v) for v in logit_pca["explained"]],
            "cumulative_explained_variance": [float(v) for v in torch.cumsum(logit_pca["explained"], dim=0)],
            "k_for_90pct": min_k_for_variance(logit_pca["explained"], 0.90),
            "k_for_95pct": min_k_for_variance(logit_pca["explained"], 0.95),
            "k_for_99pct": min_k_for_variance(logit_pca["explained"], 0.99),
        },
        "logit_projection_accuracy": logit_projection,
        "pc_centers_by_true_max": by_true_max,
    }
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    cumulative = torch.cumsum(head_pca["explained"], dim=0)
    axes[0, 0].plot(range(1, len(cumulative) + 1), cumulative.numpy(), marker="o", markersize=3)
    axes[0, 0].axhline(0.90, color="#777777", linestyle="--", linewidth=1)
    axes[0, 0].axhline(0.95, color="#777777", linestyle="--", linewidth=1)
    axes[0, 0].axhline(0.99, color="#777777", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Head-sum 64d PCA cumulative variance")
    axes[0, 0].set_xlabel("number of PCs")
    axes[0, 0].set_ylabel("cumulative variance explained")
    axes[0, 0].set_ylim(0, 1.02)

    xvals = [int(k) for k in head_projection.keys()]
    yvals = [head_projection[str(k)]["accuracy"] for k in xvals]
    axes[0, 1].plot(xvals, yvals, marker="o")
    axes[0, 1].set_title("Accuracy after projecting 64d head-sum to top-k PCs")
    axes[0, 1].set_xlabel("k PCs")
    axes[0, 1].set_ylabel("accuracy")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_xticks([0, 1, 2, 3, 4, 5, 8, 12, 16, 20, 32, 48, 64])

    # Stratified sample for PC1/PC2 scatter.
    sample_indices = []
    generator = torch.Generator().manual_seed(0)
    for max_value in range(10):
        idx = (labels == max_value).nonzero(as_tuple=False).flatten()
        take = min(len(idx), 1000)
        if take > 0:
            perm = idx[torch.randperm(len(idx), generator=generator)[:take]]
            sample_indices.append(perm)
    sample_indices_t = torch.cat(sample_indices)
    scatter = axes[1, 0].scatter(
        head_pca["scores"][sample_indices_t, 0].numpy(),
        head_pca["scores"][sample_indices_t, 1].numpy(),
        c=labels[sample_indices_t].numpy(),
        cmap="tab10",
        s=8,
        alpha=0.45,
    )
    centers_x = [by_true_max[str(i)]["pc1_mean"] for i in range(10)]
    centers_y = [by_true_max[str(i)]["pc2_mean"] for i in range(10)]
    axes[1, 0].plot(centers_x, centers_y, color="black", linewidth=1, alpha=0.7)
    for i, (x, y) in enumerate(zip(centers_x, centers_y)):
        axes[1, 0].text(x, y, str(i), ha="center", va="center", fontsize=11, weight="bold")
    axes[1, 0].set_title("Head-sum PC1/PC2 by true max")
    axes[1, 0].set_xlabel("PC1 score")
    axes[1, 0].set_ylabel("PC2 score")
    cbar = fig.colorbar(scatter, ax=axes[1, 0], ticks=range(10))
    cbar.set_label("true max")

    logit_cumulative = torch.cumsum(logit_pca["explained"], dim=0)
    logit_x = [int(k) for k in logit_projection.keys()]
    logit_y = [logit_projection[str(k)]["accuracy"] for k in logit_x]
    axes[1, 1].plot(range(1, len(logit_cumulative) + 1), logit_cumulative.numpy(), marker="o", label="logit PCA variance")
    axes[1, 1].plot(logit_x, logit_y, marker="s", label="logit projection accuracy")
    axes[1, 1].set_title("10d head-sum logits: PCA variance and accuracy")
    axes[1, 1].set_xlabel("number of PCs")
    axes[1, 1].set_ylabel("fraction")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xticks(range(0, 11))
    axes[1, 1].legend()

    fig.suptitle("Model 1: low-dimensional structure of head-sum outputs")
    fig.savefig(OUT, dpi=180)

    print(f"head_sum_only_accuracy,{full_head_acc:.6f}")
    print(
        "head_sum_k_for_variance,"
        f"90:{result['head_sum_pca']['k_for_90pct']},"
        f"95:{result['head_sum_pca']['k_for_95pct']},"
        f"99:{result['head_sum_pca']['k_for_99pct']}"
    )
    print("head_sum_projection_accuracy")
    for k in head_ks:
        print(f"{k},{head_projection[str(k)]['accuracy']:.6f},{head_projection[str(k)]['prediction_distribution']}")
    print(
        "logit_k_for_variance,"
        f"90:{result['logit_pca']['k_for_90pct']},"
        f"95:{result['logit_pca']['k_for_95pct']},"
        f"99:{result['logit_pca']['k_for_99pct']}"
    )
    print("logit_projection_accuracy")
    for k in logit_ks:
        print(f"{k},{logit_projection[str(k)]['accuracy']:.6f},{logit_projection[str(k)]['prediction_distribution']}")
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
