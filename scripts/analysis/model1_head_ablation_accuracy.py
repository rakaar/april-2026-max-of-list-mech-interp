#!/usr/bin/env python3
"""Evaluate Model 1 accuracy under head-output ablations on all 10^5 inputs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


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


def make_all_inputs(device: str):
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
    return tokens, nums.max(dim=1).values


@torch.no_grad()
def forward_ablate(model, x: torch.Tensor, ablate_heads: tuple[int, ...]):
    _, seq_len = x.shape
    positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
    h = model.tok_embed(x) + model.pos_embed(positions)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)

    layer = model.layers[0]
    head_outputs = []
    for head_idx, head in enumerate(layer.heads):
        out, _ = head(h, mask)
        if head_idx in ablate_heads:
            out = torch.zeros_like(out)
        head_outputs.append(out)

    h = h + layer.W_O(torch.cat(head_outputs, dim=-1))
    return model.unembed(h)


def evaluate(model, tokens, labels, ablate_heads: tuple[int, ...], batch_size: int = 4096):
    correct = 0
    for start in range(0, tokens.shape[0], batch_size):
        logits = forward_ablate(model, tokens[start : start + batch_size], ablate_heads)
        pred = logits[:, -1, :10].argmax(dim=-1)
        correct += int((pred == labels[start : start + batch_size]).sum())
    return correct, tokens.shape[0]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device)
    tokens, labels = make_all_inputs(device)

    ablations = [(), (0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    print("ablate_heads,accuracy,correct,total")
    for ablate in ablations:
        correct, total = evaluate(model, tokens, labels, ablate)
        label = "none" if not ablate else "+".join(str(h) for h in ablate)
        print(f"{label},{correct / total:.6f},{correct},{total}")


if __name__ == "__main__":
    main()

