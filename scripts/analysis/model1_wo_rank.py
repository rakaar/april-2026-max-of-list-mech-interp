#!/usr/bin/env python3
"""Inspect singular spectra of Model 1 W_O, per-head slices, and OV maps."""

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


def summarize(name: str, matrix: torch.Tensor) -> None:
    s = torch.linalg.svdvals(matrix)
    variance = s.square() / s.square().sum()
    top1 = float(variance[:1].sum())
    top2 = float(variance[:2].sum())
    top4 = float(variance[:4].sum())
    top8 = float(variance[:8].sum())
    top16 = float(variance[:16].sum())
    rank_1e6 = int((s > 1e-6).sum())
    rank_1e3 = int((s > 1e-3).sum())
    print(
        f"{name},{tuple(matrix.shape)},{rank_1e6},{rank_1e3},"
        f"{top1:.6f},{top2:.6f},{top4:.6f},{top8:.6f},{top16:.6f},"
        f"{float(s[0]):.6f},{float(s[1]):.6f},{float(s[2]):.6f},{float(s[3]):.6f}"
    )


def main() -> None:
    model = load_model()
    layer = model.layers[0]
    w_o = layer.W_O.weight.detach()

    print(
        "matrix,shape,rank_tol_1e-6,rank_tol_1e-3,"
        "energy_top1,energy_top2,energy_top4,energy_top8,energy_top16,"
        "s1,s2,s3,s4"
    )
    summarize("full_W_O", w_o)

    for head_idx, head in enumerate(layer.heads):
        w_v = head.W_V.weight.detach()
        d_head = w_v.shape[0]
        w_o_head = w_o[:, head_idx * d_head : (head_idx + 1) * d_head]
        ov = w_o_head @ w_v
        summarize(f"H{head_idx}_W_O_slice", w_o_head)
        summarize(f"H{head_idx}_W_V", w_v)
        summarize(f"H{head_idx}_W_O_slice_x_W_V", ov)


if __name__ == "__main__":
    main()

