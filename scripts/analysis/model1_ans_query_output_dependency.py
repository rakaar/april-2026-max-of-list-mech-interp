#!/usr/bin/env python3
"""Check whether non-ANS attention query rows affect final ANS logits in Model 1."""

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


def make_inputs(device: str):
    nums = torch.cartesian_prod(*[torch.arange(2, 10) for _ in range(5)]).to(device)
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


@torch.no_grad()
def manual_logits(model, x: torch.Tensor, mode: str):
    _, seq_len = x.shape
    positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
    resid = model.tok_embed(x) + model.pos_embed(positions)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)

    layer = model.layers[0]
    head_outputs = []
    for head in layer.heads:
        out, _ = head(resid, mask)
        head_outputs.append(out)
    concat = torch.cat(head_outputs, dim=-1)

    if mode == "zero_non_ans_query_outputs":
        concat = concat.clone()
        concat[:, :-1, :] = 0
    elif mode == "zero_ans_query_output":
        concat = concat.clone()
        concat[:, -1, :] = 0

    return model.unembed(resid + layer.W_O(concat))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model().to(device)
    nums, tokens = make_inputs(device)
    labels = nums.max(dim=1).values

    max_diff_non_ans = 0.0
    max_diff_zero_ans = 0.0
    baseline_correct = 0
    zero_non_ans_correct = 0
    zero_ans_correct = 0
    batch_size = 4096

    for start in range(0, tokens.shape[0], batch_size):
        x = tokens[start : start + batch_size]
        y = labels[start : start + batch_size]
        baseline = manual_logits(model, x, "baseline")
        zero_non_ans = manual_logits(model, x, "zero_non_ans_query_outputs")
        zero_ans = manual_logits(model, x, "zero_ans_query_output")

        max_diff_non_ans = max(
            max_diff_non_ans,
            float((baseline[:, -1, :] - zero_non_ans[:, -1, :]).abs().max()),
        )
        max_diff_zero_ans = max(
            max_diff_zero_ans,
            float((baseline[:, -1, :] - zero_ans[:, -1, :]).abs().max()),
        )
        baseline_correct += int((baseline[:, -1, :10].argmax(dim=1) == y).sum())
        zero_non_ans_correct += int((zero_non_ans[:, -1, :10].argmax(dim=1) == y).sum())
        zero_ans_correct += int((zero_ans[:, -1, :10].argmax(dim=1) == y).sum())

    total = tokens.shape[0]
    print(f"n_all_numbers_ge_2,{total}")
    print(f"zero_non_ans_query_outputs_final_ans_logits_max_abs_diff,{max_diff_non_ans:.6f}")
    print(f"zero_ans_query_output_final_ans_logits_max_abs_diff,{max_diff_zero_ans:.6f}")
    print(f"baseline_accuracy,{baseline_correct / total:.6f},{baseline_correct}/{total}")
    print(
        f"zero_non_ans_query_outputs_accuracy,"
        f"{zero_non_ans_correct / total:.6f},{zero_non_ans_correct}/{total}"
    )
    print(f"zero_ans_query_output_accuracy,{zero_ans_correct / total:.6f},{zero_ans_correct}/{total}")


if __name__ == "__main__":
    main()

