#!/usr/bin/env python3
"""Measure per-head ANS output cosine with the true-max unembedding."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_head_output_target_unembed_cosine.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_head_output_target_unembed_cosine.json"
BATCH_SIZE = 4096
EXPECTED_COUNTS_BY_MAX = {
    0: 1,
    1: 31,
    2: 211,
    3: 781,
    4: 2101,
    5: 4651,
    6: 9031,
    7: 15961,
    8: 26281,
    9: 40951,
}


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


def make_inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nums = torch.cartesian_prod(*[torch.arange(10) for _ in range(5)]).to(device)
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
    return nums, tokens


def plot_heatmaps(mean_cosine: torch.Tensor, std_cosine: torch.Tensor) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.0), constrained_layout=True)
    heads = [f"H{i}" for i in range(mean_cosine.shape[1])]
    max_values = list(range(10))

    mean_abs = max(abs(float(mean_cosine.min())), abs(float(mean_cosine.max())), 1e-6)
    panels = [
        (
            mean_cosine,
            "Mean cosine",
            "coolwarm",
            TwoSlopeNorm(vmin=-mean_abs, vcenter=0.0, vmax=mean_abs),
        ),
        (std_cosine, "Unbiased std cosine", "magma", None),
    ]
    for ax, (matrix, title, cmap, norm) in zip(axes, panels):
        im = ax.imshow(matrix.numpy(), cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(heads)), heads)
        ax.set_yticks(max_values, max_values)
        ax.set_xlabel("Head")
        ax.set_ylabel("True max")
        for y in max_values:
            for x in range(len(heads)):
                value = float(matrix[y, x])
                ax.text(
                    x,
                    y,
                    f"{value:.1e}" if title.startswith("Unbiased") else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7 if title.startswith("Unbiased") else 8,
                    color="white" if title.startswith("Unbiased") and value < 2.0e-4 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Model 1: head output direction vs true-max unembedding")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    nums, tokens = make_inputs(device)
    labels = nums.max(dim=1).values

    n_heads = model.n_heads
    counts = torch.bincount(labels, minlength=10).cpu()
    expected_counts = torch.tensor([EXPECTED_COUNTS_BY_MAX[i] for i in range(10)])
    if not torch.equal(counts, expected_counts):
        raise ValueError(f"unexpected true-max counts: {counts.tolist()}")

    cosine_sum = torch.zeros(10, n_heads, dtype=torch.float64)
    cosine_sq_sum = torch.zeros(10, n_heads, dtype=torch.float64)
    min_cosine = torch.full((n_heads,), float("inf"), dtype=torch.float64)
    max_cosine = torch.full((n_heads,), float("-inf"), dtype=torch.float64)

    layer = model.layers[0]
    w_o = layer.W_O.weight.detach()
    seq_len = tokens.shape[1]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)

    with torch.no_grad():
        for start in range(0, tokens.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, tokens.shape[0])
            batch_tokens = tokens[start:end]
            batch_labels = labels[start:end]
            batch_labels_cpu = batch_labels.cpu()
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            resid = model.tok_embed(batch_tokens) + model.pos_embed(positions)
            target_u = model.unembed.weight.detach()[batch_labels]

            for head_idx, head in enumerate(layer.heads):
                head_values, _ = head(resid, causal_mask)
                d_head = head.d_head
                w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
                head_vec = head_values[:, 10, :] @ w_o_head.T
                cosine = F.cosine_similarity(head_vec, target_u, dim=-1)
                cosine_cpu = cosine.cpu().to(torch.float64)
                min_cosine[head_idx] = torch.minimum(min_cosine[head_idx], cosine_cpu.min())
                max_cosine[head_idx] = torch.maximum(max_cosine[head_idx], cosine_cpu.max())

                for max_value in range(10):
                    mask = batch_labels_cpu == max_value
                    if not bool(mask.any()):
                        continue
                    cosine_sum[max_value, head_idx] += cosine_cpu[mask].sum()
                    cosine_sq_sum[max_value, head_idx] += cosine_cpu[mask].square().sum()

    counts_f = counts.to(torch.float64).unsqueeze(1)
    mean_cosine = cosine_sum / counts_f
    numerator = cosine_sq_sum - counts_f * mean_cosine.square()
    denominator = (counts_f - 1).clamp_min(1.0)
    variance = (numerator / denominator).clamp_min(0.0)
    variance[counts <= 1] = 0.0
    std_cosine = variance.sqrt()

    data = {
        "description": (
            "All 10^5 Model 1 inputs. For each head h, Hh_vec is the actual "
            "ANS-position head output after that head's W_O slice. The reported "
            "metric is cosine_similarity(Hh_vec, W_U[true_max])."
        ),
        "n_inputs_total": int(tokens.shape[0]),
        "sequence_format": "[BOS] n0 [SEP] n1 [SEP] n2 [SEP] n3 [SEP] n4 [ANS]",
        "heads": [f"H{i}" for i in range(n_heads)],
        "counts_by_true_max": {str(i): int(counts[i]) for i in range(10)},
        "cosine_range_observed_by_head": {
            f"H{head_idx}": {
                "min": float(min_cosine[head_idx]),
                "max": float(max_cosine[head_idx]),
            }
            for head_idx in range(n_heads)
        },
        "rows": [
            {
                "true_max": max_value,
                "count": int(counts[max_value]),
                "mean_cosine": {
                    f"H{head_idx}": float(mean_cosine[max_value, head_idx])
                    for head_idx in range(n_heads)
                },
                "std_cosine": {
                    f"H{head_idx}": float(std_cosine[max_value, head_idx])
                    for head_idx in range(n_heads)
                },
            }
            for max_value in range(10)
        ],
    }

    JSON_OUT.write_text(json.dumps(data, indent=2) + "\n")
    plot_heatmaps(mean_cosine, std_cosine)

    print("true_max,count," + ",".join(f"H{h}_mean" for h in range(n_heads)) + "," + ",".join(f"H{h}_std" for h in range(n_heads)))
    for max_value in range(10):
        means = ",".join(f"{float(mean_cosine[max_value, h]):.6f}" for h in range(n_heads))
        stds = ",".join(f"{float(std_cosine[max_value, h]):.6f}" for h in range(n_heads))
        print(f"{max_value},{int(counts[max_value])},{means},{stds}")
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
