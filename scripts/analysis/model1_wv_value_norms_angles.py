#!/usr/bin/env python3
"""Inspect norms and pairwise angles of W_V-applied number embeddings."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_wv_value_norms_angles.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_wv_value_norms_angles.json"


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


def pairwise_angles_degrees(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unit = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosine = (unit @ unit.T).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine))
    return cosine, angles


def main() -> None:
    model = load_model()
    number_embeddings = model.tok_embed.weight.detach()[:10]
    numbers = torch.arange(10, dtype=torch.float32)
    offdiag_mask = ~torch.eye(10, dtype=torch.bool)

    results = []
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)

    for head_idx, head in enumerate(model.layers[0].heads):
        value_vectors = number_embeddings @ head.W_V.weight.detach().T
        norms = value_vectors.norm(dim=1)
        cosine, angles = pairwise_angles_degrees(value_vectors)
        adjacent_angles = torch.tensor([angles[i, i + 1] for i in range(9)])
        norm_corr = torch.corrcoef(torch.stack([numbers, norms]))[0, 1]
        norm_order_desc = [int(i) for i in torch.argsort(norms, descending=True)]
        smallest_angle = float(angles[offdiag_mask].min())
        largest_angle = float(angles[offdiag_mask].max())

        results.append(
            {
                "head": head_idx,
                "definition": "value_vectors[n] = E[n] @ W_V_h.T, token-only, no position embedding",
                "norms": [float(x) for x in norms],
                "norm_corr_with_number": float(norm_corr),
                "norm_order_desc": norm_order_desc,
                "largest_norm_number": int(norm_order_desc[0]),
                "smallest_norm_number": int(norm_order_desc[-1]),
                "pairwise_cosine": [[float(x) for x in row] for row in cosine],
                "pairwise_angle_degrees": [[float(x) for x in row] for row in angles],
                "adjacent_angle_degrees": [float(x) for x in adjacent_angles],
                "mean_adjacent_angle_degrees": float(adjacent_angles.mean()),
                "mean_offdiag_angle_degrees": float(angles[offdiag_mask].mean()),
                "min_offdiag_angle_degrees": smallest_angle,
                "max_offdiag_angle_degrees": largest_angle,
            }
        )

        ax = axes[0, head_idx]
        ax.plot(range(10), norms.numpy(), color="#111827", marker="o", linewidth=1.6)
        ax.bar(range(10), norms.numpy(), color="#2563eb", alpha=0.45)
        ax.set_xticks(range(10))
        ax.set_xlabel("Number token")
        if head_idx == 0:
            ax.set_ylabel("||E[n] @ W_V_h.T||")
        ax.set_title(
            f"H{head_idx} value norms\n"
            f"corr={float(norm_corr):+.3f}, largest={norm_order_desc[0]}"
        )
        ax.grid(axis="y", alpha=0.25)

        ax = axes[1, head_idx]
        im = ax.imshow(angles.numpy(), cmap="magma_r", vmin=0, vmax=180)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Number token")
        if head_idx == 0:
            ax.set_ylabel("Number token")
        ax.set_title(
            f"H{head_idx} pairwise angles\n"
            f"mean offdiag={float(angles[offdiag_mask].mean()):.1f} deg"
        )
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                ax.text(
                    j,
                    i,
                    f"{float(angles[i, j]):.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if float(angles[i, j]) > 90 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Model 1: value-vector norms and angles for E[number] @ W_V_h.T",
        fontsize=15,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    JSON_OUT.write_text(json.dumps(results, indent=2) + "\n")

    print("head,norm_corr,largest_norm,smallest_norm,mean_adjacent_angle,mean_offdiag_angle,min_angle,max_angle,norm_order_desc")
    for row in results:
        print(
            f"{row['head']},"
            f"{row['norm_corr_with_number']:+.6f},"
            f"{row['largest_norm_number']},"
            f"{row['smallest_norm_number']},"
            f"{row['mean_adjacent_angle_degrees']:.6f},"
            f"{row['mean_offdiag_angle_degrees']:.6f},"
            f"{row['min_offdiag_angle_degrees']:.6f},"
            f"{row['max_offdiag_angle_degrees']:.6f},"
            + "-".join(str(x) for x in row["norm_order_desc"])
        )
    print("norms_by_head")
    for row in results:
        print(
            f"H{row['head']},"
            + ",".join(f"{value:.6f}" for value in row["norms"])
        )
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
