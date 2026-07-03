#!/usr/bin/env python3
"""Decompose target-logit contributions for selected max-6 and max-7 examples."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_target_logit_contribution_examples.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_target_logit_contribution_examples.json"

EXAMPLES = [
    [6, 2, 3, 4, 5],
    [2, 6, 4, 5, 3],
    [2, 3, 6, 4, 5],
    [5, 4, 3, 2, 6],
    [6, 6, 5, 4, 3],
    [7, 2, 3, 4, 5],
    [2, 7, 4, 6, 3],
    [2, 3, 7, 4, 6],
    [6, 5, 4, 3, 7],
    [7, 7, 6, 5, 4],
]

COMPONENTS = ["residual", "H0", "H1", "H2", "H3"]
PLOT_COMPONENTS = COMPONENTS + ["final"]


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


def main() -> None:
    model = load_model()
    tokens = torch.tensor([tokenize(nums) for nums in EXAMPLES], dtype=torch.long)
    nums_t = torch.tensor(EXAMPLES, dtype=torch.long)
    labels = nums_t.max(dim=1).values

    with torch.no_grad():
        model_logits, _ = model(tokens)
        batch, seq_len = tokens.shape
        positions = torch.arange(seq_len).unsqueeze(0)
        resid = model.tok_embed(tokens) + model.pos_embed(positions)
        ans_resid = resid[:, 10, :]
        layer = model.layers[0]
        w_o = layer.W_O.weight.detach()
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

        head_vectors = []
        for head_idx, head in enumerate(layer.heads):
            head_values, _ = head(resid, mask)
            d_head = head.d_head
            w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
            head_vectors.append(head_values[:, 10, :] @ w_o_head.T)

        component_vectors = {
            "residual": ans_resid,
            "H0": head_vectors[0],
            "H1": head_vectors[1],
            "H2": head_vectors[2],
            "H3": head_vectors[3],
        }
        final_vec = ans_resid + sum(head_vectors)
        component_vectors["final"] = final_vec

        component_logits = {
            name: number_logits(vec, model) for name, vec in component_vectors.items()
        }

    raw_target = torch.empty((len(PLOT_COMPONENTS), batch))
    signed_share = torch.empty((len(COMPONENTS), batch))
    positive_share = torch.empty((len(COMPONENTS), batch))

    examples = []
    for idx, nums in enumerate(EXAMPLES):
        target = int(labels[idx])
        final_target = float(component_logits["final"][idx, target])
        component_values = torch.tensor(
            [float(component_logits[name][idx, target]) for name in COMPONENTS]
        )
        positive = component_values.clamp_min(0)
        positive_total = float(positive.sum())

        for row_idx, name in enumerate(PLOT_COMPONENTS):
            raw_target[row_idx, idx] = float(component_logits[name][idx, target])
        for row_idx, name in enumerate(COMPONENTS):
            value = float(component_logits[name][idx, target])
            signed_share[row_idx, idx] = value / final_target if abs(final_target) > 1e-12 else float("nan")
            positive_share[row_idx, idx] = (
                max(value, 0.0) / positive_total if positive_total > 1e-12 else float("nan")
            )

        examples.append(
            {
                "example_index": idx,
                "nums": nums,
                "target_logit": target,
                "model_pred": int(model_logits[idx, 10, :10].argmax()),
                "raw_target_logit_contribution": {
                    name: float(component_logits[name][idx, target]) for name in PLOT_COMPONENTS
                },
                "signed_share_of_final_target_logit": {
                    name: float(signed_share[row_idx, idx])
                    for row_idx, name in enumerate(COMPONENTS)
                },
                "positive_share_of_positive_target_logit_contributions": {
                    name: float(positive_share[row_idx, idx])
                    for row_idx, name in enumerate(COMPONENTS)
                },
            }
        )

    averages = {}
    for target in [6, 7]:
        indices = [idx for idx, label in enumerate(labels.tolist()) if label == target]
        averages[str(target)] = {
            "raw_target_logit_contribution": {
                name: float(torch.tensor([component_logits[name][idx, target] for idx in indices]).mean())
                for name in PLOT_COMPONENTS
            },
            "signed_share_of_final_target_logit": {
                name: float(torch.stack([signed_share[row_idx, idx] for idx in indices]).mean())
                for row_idx, name in enumerate(COMPONENTS)
            },
            "positive_share_of_positive_target_logit_contributions": {
                name: float(torch.stack([positive_share[row_idx, idx] for idx in indices]).mean())
                for row_idx, name in enumerate(COMPONENTS)
            },
        }

    result = {
        "description": (
            "For max-6 examples, target logit is 6. For max-7 examples, target logit is 7. "
            "Raw contributions are additive. Signed shares can be negative or exceed 1 due to "
            "cancellation. Positive shares sum to 1 across positive contributors only."
        ),
        "components": COMPONENTS,
        "examples": examples,
        "averages_by_target_logit": averages,
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    labels_x = [f"{i}: {nums}\ntarget={int(labels[i])}" for i, nums in enumerate(EXAMPLES)]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

    vmax = max(abs(float(raw_target.min())), abs(float(raw_target.max())))
    im = axes[0].imshow(raw_target.numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_title("Raw contribution to target logit")
    axes[0].set_yticks(range(len(PLOT_COMPONENTS)))
    axes[0].set_yticklabels(PLOT_COMPONENTS)
    axes[0].set_xticks(range(batch))
    axes[0].set_xticklabels(labels_x, rotation=35, ha="right", fontsize=8)
    for y in range(len(PLOT_COMPONENTS)):
        for x in range(batch):
            value = float(raw_target[y, x])
            axes[0].text(
                x,
                y,
                f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 * vmax else "black",
            )
    fig.colorbar(im, ax=axes[0], fraction=0.025, pad=0.02, label="target logit contribution")

    im = axes[1].imshow(100 * signed_share.numpy(), cmap="coolwarm", vmin=-150, vmax=150, aspect="auto")
    axes[1].set_title("Signed share of final target logit (%)")
    axes[1].set_yticks(range(len(COMPONENTS)))
    axes[1].set_yticklabels(COMPONENTS)
    axes[1].set_xticks(range(batch))
    axes[1].set_xticklabels(labels_x, rotation=35, ha="right", fontsize=8)
    for y in range(len(COMPONENTS)):
        for x in range(batch):
            value = float(100 * signed_share[y, x])
            axes[1].text(x, y, f"{value:+.0f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.025, pad=0.02, label="% of final target logit")

    im = axes[2].imshow(100 * positive_share.numpy(), cmap="viridis", vmin=0, vmax=100, aspect="auto")
    axes[2].set_title("Positive share among positive target-logit contributors (%)")
    axes[2].set_yticks(range(len(COMPONENTS)))
    axes[2].set_yticklabels(COMPONENTS)
    axes[2].set_xticks(range(batch))
    axes[2].set_xticklabels(labels_x, rotation=35, ha="right", fontsize=8)
    for y in range(len(COMPONENTS)):
        for x in range(batch):
            value = float(100 * positive_share[y, x])
            axes[2].text(
                x,
                y,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 55 else "black",
            )
    fig.colorbar(im, ax=axes[2], fraction=0.025, pad=0.02, label="% of positive contributions")

    fig.suptitle("Model 1: target-logit contribution decomposition for max 6 vs max 7 examples")
    fig.savefig(OUT, dpi=180)

    print("example,target,pred,component,raw_target_logit,signed_share,positive_share")
    for item in examples:
        for name in COMPONENTS:
            print(
                f"{item['example_index']},{item['target_logit']},{item['model_pred']},{name},"
                f"{item['raw_target_logit_contribution'][name]:+.6f},"
                f"{item['signed_share_of_final_target_logit'][name]:+.6f},"
                f"{item['positive_share_of_positive_target_logit_contributions'][name]:+.6f}"
            )
    print("averages_by_target")
    for target, values in averages.items():
        raw = values["raw_target_logit_contribution"]
        pos = values["positive_share_of_positive_target_logit_contributions"]
        print(
            f"target={target},raw="
            + ",".join(f"{name}:{raw[name]:+.6f}" for name in PLOT_COMPONENTS)
        )
        print(
            f"target={target},positive_share="
            + ",".join(f"{name}:{pos[name]:+.6f}" for name in COMPONENTS)
        )
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
