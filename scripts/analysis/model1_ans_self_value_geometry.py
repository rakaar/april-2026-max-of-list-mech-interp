#!/usr/bin/env python3
"""Analyze the per-head [ANS]->[ANS] value and output-write vectors."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_ans_self_value_geometry.png"
JSON_OUT = ROOT / "docs" / "assets" / "model1_ans_self_value_geometry.json"
HEADS = ["H0", "H1", "H2", "H3"]


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


def cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    x_norm = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return x_norm @ x_norm.T


def angle_matrix_deg(cos: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def svd_energy(x: torch.Tensor) -> dict[str, list[float]]:
    _, singular_values, _ = torch.linalg.svd(x, full_matrices=False)
    energy = singular_values.square()
    return {
        "singular_values": [float(v) for v in singular_values],
        "energy_fraction": [float(v) for v in (energy / energy.sum())],
    }


def main() -> None:
    model = load_model()
    layer = model.layers[0]
    w_o = layer.W_O.weight.detach()
    ans_token = torch.tensor([12], dtype=torch.long)
    ans_position = torch.tensor([10], dtype=torch.long)

    with torch.no_grad():
        ans_resid = model.tok_embed(ans_token) + model.pos_embed(ans_position)
        value_vectors = []
        output_vectors = []
        for head_idx, head in enumerate(layer.heads):
            value = head.W_V(ans_resid)
            d_head = head.d_head
            w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
            output = value @ w_o_head.T
            value_vectors.append(value.squeeze(0))
            output_vectors.append(output.squeeze(0))

        value_matrix = torch.stack(value_vectors)
        output_matrix = torch.stack(output_vectors)
        digit_logits = output_matrix @ model.unembed.weight.detach()[:10].T

    value_cos = cosine_matrix(value_matrix)
    output_cos = cosine_matrix(output_matrix)
    value_angles = angle_matrix_deg(value_cos)
    output_angles = angle_matrix_deg(output_cos)

    result = {
        "description": (
            "For each head, the source vector is the actual [ANS] residual at position 10: "
            "E[ANS] + P[10]. value_vector_h = source @ W_V_h.T. "
            "output_vector_h = value_vector_h @ W_O_h.T."
        ),
        "heads": HEADS,
        "value_shape": list(value_matrix.shape),
        "output_shape": list(output_matrix.shape),
        "value_vectors": value_matrix.tolist(),
        "output_vectors": output_matrix.tolist(),
        "value_norms": [float(v) for v in value_matrix.norm(dim=1)],
        "output_norms": [float(v) for v in output_matrix.norm(dim=1)],
        "value_pairwise_cosine": value_cos.tolist(),
        "value_pairwise_angle_degrees": value_angles.tolist(),
        "output_pairwise_cosine": output_cos.tolist(),
        "output_pairwise_angle_degrees": output_angles.tolist(),
        "value_svd": svd_energy(value_matrix),
        "output_svd": svd_energy(output_matrix),
        "digit_logit_effects": digit_logits.tolist(),
        "digit_logit_argmax": [int(x) for x in digit_logits.argmax(dim=1)],
        "digit_logit_argmin": [int(x) for x in digit_logits.argmin(dim=1)],
    }
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)

    im = axes[0, 0].imshow(value_matrix.numpy(), cmap="coolwarm", aspect="auto")
    axes[0, 0].set_title("[ANS] value vectors: 4 heads x 16 dims")
    axes[0, 0].set_yticks(range(4))
    axes[0, 0].set_yticklabels(HEADS)
    axes[0, 0].set_xlabel("value dimension")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(value_cos.numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axes[0, 1].set_title("Value-vector cosine")
    axes[0, 1].set_xticks(range(4))
    axes[0, 1].set_xticklabels(HEADS)
    axes[0, 1].set_yticks(range(4))
    axes[0, 1].set_yticklabels(HEADS)
    for y in range(4):
        for x in range(4):
            axes[0, 1].text(x, y, f"{float(value_cos[y, x]):+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    x = torch.arange(4)
    axes[0, 2].bar(x - 0.17, value_matrix.norm(dim=1).numpy(), width=0.34, label="value 16d")
    axes[0, 2].bar(x + 0.17, output_matrix.norm(dim=1).numpy(), width=0.34, label="output 64d")
    axes[0, 2].set_title("Vector norms")
    axes[0, 2].set_xticks(range(4))
    axes[0, 2].set_xticklabels(HEADS)
    axes[0, 2].legend()
    axes[0, 2].set_ylabel("L2 norm")

    im = axes[1, 0].imshow(output_matrix.numpy(), cmap="coolwarm", aspect="auto")
    axes[1, 0].set_title("Post-W_O output vectors: 4 heads x 64 dims")
    axes[1, 0].set_yticks(range(4))
    axes[1, 0].set_yticklabels(HEADS)
    axes[1, 0].set_xlabel("residual dimension")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(output_cos.numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 1].set_title("Output-vector cosine")
    axes[1, 1].set_xticks(range(4))
    axes[1, 1].set_xticklabels(HEADS)
    axes[1, 1].set_yticks(range(4))
    axes[1, 1].set_yticklabels(HEADS)
    for y in range(4):
        for x in range(4):
            axes[1, 1].text(x, y, f"{float(output_cos[y, x]):+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    vmax = max(abs(float(digit_logits.min())), abs(float(digit_logits.max())))
    im = axes[1, 2].imshow(digit_logits.numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1, 2].set_title("Digit-logit effects of [ANS] self write")
    axes[1, 2].set_yticks(range(4))
    axes[1, 2].set_yticklabels(HEADS)
    axes[1, 2].set_xticks(range(10))
    axes[1, 2].set_xticklabels(range(10))
    axes[1, 2].set_xlabel("output digit")
    for y in range(4):
        for xidx in range(10):
            value = float(digit_logits[y, xidx])
            axes[1, 2].text(
                xidx,
                y,
                f"{value:+.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 * vmax else "black",
            )
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle("Model 1: geometry of [ANS] one-hot self value/write by head")
    fig.savefig(OUT, dpi=180)

    print("head,value_norm,output_norm,logit_argmax,logit_argmin,logits0to9")
    for i, head in enumerate(HEADS):
        print(
            head,
            f"{float(value_matrix[i].norm()):.6f}",
            f"{float(output_matrix[i].norm()):.6f}",
            int(digit_logits[i].argmax()),
            int(digit_logits[i].argmin()),
            ",".join(f"{float(v):+.6f}" for v in digit_logits[i]),
            sep=",",
        )
    print("value_cosine")
    for row in value_cos:
        print(",".join(f"{float(v):+.6f}" for v in row))
    print("output_cosine")
    for row in output_cos:
        print(",".join(f"{float(v):+.6f}" for v in row))
    print("value_energy_fraction", ",".join(f"{v:.6f}" for v in result["value_svd"]["energy_fraction"]))
    print("output_energy_fraction", ",".join(f"{v:.6f}" for v in result["output_svd"]["energy_fraction"]))
    print(f"wrote,{OUT}")
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
