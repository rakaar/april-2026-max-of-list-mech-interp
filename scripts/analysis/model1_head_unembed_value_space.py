#!/usr/bin/env python3
"""Analyze number unembedding directions through each head's output/value maps."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "docs" / "assets"
PCA_OUT = ASSET_DIR / "model1_head_unembed_value_space_pca.png"
HEATMAP_OUT = ASSET_DIR / "model1_head_ov_number_logit_effect.png"


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


def pca_rows(x: torch.Tensor, k: int = 3):
    """PCA scores for row observations."""
    centered = x - x.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    scores = u[:, :k] * s[:k]
    energy = s.square() / s.square().sum()
    return scores, energy


def is_monotonic(x: torch.Tensor) -> bool:
    return bool(
        all(x[i] <= x[i + 1] for i in range(len(x) - 1))
        or all(x[i] >= x[i + 1] for i in range(len(x) - 1))
    )


def annotate_pca(ax, scores: torch.Tensor, title: str) -> None:
    xs = scores[:, 0].numpy()
    ys = scores[:, 1].numpy()
    ax.plot(xs, ys, color="#9ca3af", linewidth=1, alpha=0.7)
    ax.scatter(xs, ys, c=range(10), cmap="viridis", s=48, zorder=3)
    for token, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(str(token), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#d1d5db", linewidth=0.8)
    ax.axvline(0, color="#d1d5db", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def main() -> None:
    model = load_model()
    layer = model.layers[0]
    e_numbers = model.tok_embed.weight.detach()[:10]
    u_numbers = model.unembed.weight.detach()[:10]
    w_o = layer.W_O.weight.detach()
    token_values = torch.arange(10, dtype=torch.float32)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig_pca, pca_axes = plt.subplots(2, 4, figsize=(16, 7))
    fig_heat, heat_axes = plt.subplots(1, 4, figsize=(16, 3.7))

    print(
        "head,space,pc1,pc2,pc3,pc1_token_corr,pc1_monotonic,"
        "norm_token_corr,norm_monotonic,largest_norm_token,smallest_norm_token,order_pc1_asc"
    )

    for head_idx, head in enumerate(layer.heads):
        w_v = head.W_V.weight.detach()
        d_head = w_v.shape[0]
        w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]

        # Columns are output-number unembedding directions pulled into this
        # head's 16d value space.
        value_read_dirs = w_o_head.T @ u_numbers.T  # 16 x 10
        value_read_observations = value_read_dirs.T  # 10 x 16

        # Direct source-number token to output-number logit effect if this head
        # attends completely to that source token.
        ov_logit_effect = e_numbers @ w_v.T @ w_o_head.T @ u_numbers.T  # 10 x 10

        for row_idx, (space_name, observations) in enumerate(
            [
                ("W_O^T U", value_read_observations),
                ("E OV U", ov_logit_effect),
            ]
        ):
            scores, ratios = pca_rows(observations)
            norms = observations.norm(dim=1)
            pc1_corr = torch.corrcoef(torch.stack([token_values, scores[:, 0]]))[0, 1]
            norm_corr = torch.corrcoef(torch.stack([token_values, norms]))[0, 1]
            order_pc1_asc = "-".join(str(int(i)) for i in torch.argsort(scores[:, 0]))
            print(
                f"{head_idx},{space_name},"
                f"{float(ratios[0]):.6f},{float(ratios[1]):.6f},{float(ratios[2]):.6f},"
                f"{float(pc1_corr):.6f},{is_monotonic(scores[:, 0])},"
                f"{float(norm_corr):.6f},{is_monotonic(norms)},"
                f"{int(norms.argmax())},{int(norms.argmin())},{order_pc1_asc}"
            )

            title = (
                f"H{head_idx} {space_name}: "
                f"PC1 {100 * float(ratios[0]):.1f}%, "
                f"PC1+2 {100 * float(ratios[:2].sum()):.1f}%"
            )
            annotate_pca(pca_axes[row_idx, head_idx], scores, title)

        row_argmax = ov_logit_effect.argmax(dim=1)
        diag_ranks = []
        for source in range(10):
            descending = torch.argsort(ov_logit_effect[source], descending=True)
            rank = int((descending == source).nonzero(as_tuple=True)[0].item()) + 1
            diag_ranks.append(rank)
        print(
            f"head {head_idx} E_OV_U row_argmax="
            f"{','.join(str(int(x)) for x in row_argmax)} "
            f"self_logit_ranks={','.join(str(x) for x in diag_ranks)}"
        )

        ax = heat_axes[head_idx]
        im = ax.imshow(ov_logit_effect.numpy(), cmap="coolwarm")
        ax.set_title(f"H{head_idx} E OV U")
        ax.set_xlabel("Output number logit")
        ax.set_ylabel("Source number token")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        fig_heat.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig_pca.suptitle("Model 1 per-head number unembedding directions in value/logit space", y=0.995)
    fig_pca.tight_layout()
    fig_pca.savefig(PCA_OUT, dpi=180)

    fig_heat.suptitle("Model 1 direct per-head source-number to output-number logit effect", y=1.05)
    fig_heat.tight_layout()
    fig_heat.savefig(HEATMAP_OUT, dpi=180)

    print(f"wrote,{PCA_OUT}")
    print(f"wrote,{HEATMAP_OUT}")


if __name__ == "__main__":
    main()

