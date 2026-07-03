#!/usr/bin/env python3
"""Plot per-head source-number OV effects on output-number logits."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_head_ov_logit_effects.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_head_ov_logit_effects.json"
NUMBER_POSITIONS = torch.tensor([1, 3, 5, 7, 9])


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


def head_logit_effects(model):
    layer = model.layers[0]
    embeddings = model.tok_embed.weight.detach()
    positions = model.pos_embed.weight.detach()
    e_numbers = embeddings[:10]
    u_numbers = model.unembed.weight.detach()[:10]
    w_o = layer.W_O.weight.detach()

    results = []
    for head_idx, head in enumerate(layer.heads):
        d_head = head.d_head
        w_v = head.W_V.weight.detach()
        w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]

        token_only = e_numbers @ w_v.T @ w_o_head.T @ u_numbers.T

        full_by_position = []
        for pos in NUMBER_POSITIONS.tolist():
            source_resid = e_numbers + positions[pos]
            full_by_position.append(source_resid @ w_v.T @ w_o_head.T @ u_numbers.T)
        full_by_position_t = torch.stack(full_by_position)
        position_mean = full_by_position_t.mean(dim=0)
        position_std = full_by_position_t.std(dim=0)

        results.append(
            {
                "head": head_idx,
                "token_only_effect": [[float(x) for x in row] for row in token_only],
                "position_mean_effect": [[float(x) for x in row] for row in position_mean],
                "position_std_effect": [[float(x) for x in row] for row in position_std],
                "token_only_row_argmax": [int(x) for x in token_only.argmax(dim=1)],
                "position_mean_row_argmax": [int(x) for x in position_mean.argmax(dim=1)],
                "token_only_self_logit_rank": self_ranks(token_only),
                "position_mean_self_logit_rank": self_ranks(position_mean),
                "token_only_row_max_value": [float(x) for x in token_only.max(dim=1).values],
                "position_mean_row_max_value": [float(x) for x in position_mean.max(dim=1).values],
            }
        )
    return results


def self_ranks(matrix: torch.Tensor) -> list[int]:
    ranks = []
    for source in range(10):
        descending = torch.argsort(matrix[source], descending=True)
        ranks.append(int((descending == source).nonzero(as_tuple=True)[0].item()) + 1)
    return ranks


def heatmap(ax, matrix: torch.Tensor, title: str, show_ylabel: bool) -> None:
    values = matrix.numpy()
    vmax = max(abs(float(values.min())), abs(float(values.max())), 1e-6)
    im = ax.imshow(values, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Output digit logit")
    if show_ylabel:
        ax.set_ylabel("Attended source digit")
    else:
        ax.set_ylabel("")
    for source in range(10):
        argmax = int(matrix[source].argmax())
        ax.scatter(argmax, source, marker="s", s=52, facecolors="none", edgecolors="black", linewidths=1.2)
    return im


def main() -> None:
    model = load_model()
    results = head_logit_effects(model)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    for head_idx, row in enumerate(results):
        token_only = torch.tensor(row["token_only_effect"])
        position_mean = torch.tensor(row["position_mean_effect"])
        im = heatmap(
            axes[0, head_idx],
            token_only,
            f"H{head_idx} token-only\nrow argmax={row['token_only_row_argmax']}",
            show_ylabel=head_idx == 0,
        )
        fig.colorbar(im, ax=axes[0, head_idx], fraction=0.046, pad=0.04)
        im = heatmap(
            axes[1, head_idx],
            position_mean,
            f"H{head_idx} mean over positions 1,3,5,7,9\nrow argmax={row['position_mean_row_argmax']}",
            show_ylabel=head_idx == 0,
        )
        fig.colorbar(im, ax=axes[1, head_idx], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Model 1: per-head one-hot source-number OV effects on output-number logits",
        fontsize=15,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    JSON_OUT.write_text(json.dumps(results, indent=2) + "\n")

    print("head,variant,row_argmax,self_logit_ranks")
    for row in results:
        print(
            f"{row['head']},token_only,"
            + "-".join(str(x) for x in row["token_only_row_argmax"])
            + ","
            + "-".join(str(x) for x in row["token_only_self_logit_rank"])
        )
        print(
            f"{row['head']},position_mean,"
            + "-".join(str(x) for x in row["position_mean_row_argmax"])
            + ","
            + "-".join(str(x) for x in row["position_mean_self_logit_rank"])
        )
    print("selected_recruited_reads_token_only")
    for head_idx, sources in [(3, range(2, 10)), (2, range(7, 10)), (0, [9])]:
        row = results[head_idx]
        for source in sources:
            argmax = row["token_only_row_argmax"][source]
            max_value = row["token_only_row_max_value"][source]
            print(f"H{head_idx},source={source},argmax={argmax},max_logit_effect={max_value:.6f}")
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
