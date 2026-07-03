#!/usr/bin/env python3
"""Plot post-softmax Model 1 attention matrices for random examples."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "model1_random_attention_examples.png"
EXAMPLES_OUT = ROOT / "docs" / "assets" / "model1_random_attention_examples.json"

SPECIAL = {10: "BOS", 11: "SEP", 12: "ANS", 13: "EOS"}
SEED = 20260629


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


def labels_for(tokens: list[int]) -> list[str]:
    return [f"{SPECIAL.get(tok, str(tok))}@{pos}" for pos, tok in enumerate(tokens)]


def main() -> None:
    generator = torch.Generator().manual_seed(SEED)
    examples = torch.randint(0, 10, (5, 5), generator=generator).tolist()

    model = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    token_batches = [tokenize(nums) for nums in examples]
    x = torch.tensor(token_batches, dtype=torch.long, device=device)

    with torch.no_grad():
        logits, attention_patterns = model(x)

    # Model 1 has one layer. Shape: batch, heads, query_pos, key_pos.
    attn = attention_patterns[0].detach().cpu()
    preds = logits[:, -1, :10].argmax(dim=-1).detach().cpu().tolist()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_OUT.write_text(
        json.dumps(
            [
                {"nums": nums, "max": max(nums), "pred": pred, "tokens": tokens}
                for nums, pred, tokens in zip(examples, preds, token_batches)
            ],
            indent=2,
        )
        + "\n"
    )

    fig, axes = plt.subplots(5, 4, figsize=(23, 25), constrained_layout=True)
    for example_idx, (nums, tokens, pred) in enumerate(zip(examples, token_batches, preds)):
        token_labels = labels_for(tokens)
        for head_idx in range(4):
            ax = axes[example_idx, head_idx]
            ax.imshow(attn[example_idx, head_idx].numpy(), cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(token_labels)))
            ax.set_yticks(range(len(token_labels)))
            ax.set_xticklabels(token_labels, rotation=45, ha="left", fontsize=7)
            ax.set_yticklabels(token_labels, fontsize=7)
            ax.xaxis.tick_top()
            ax.tick_params(axis="both", length=0)
            ax.set_title(
                f"ex{example_idx + 1} H{head_idx}: {nums} max={max(nums)} pred={pred}",
                fontsize=10,
            )
            if head_idx == 0:
                ax.set_ylabel("Query position", fontsize=9)
            ax.set_xlabel("Key position", fontsize=9)

            # Highlight the ANS query row because this row drives the prediction.
            ans_pos = len(tokens) - 1
            ax.axhline(ans_pos - 0.5, color="#f97316", linewidth=1.0)
            ax.axhline(ans_pos + 0.5, color="#f97316", linewidth=1.0)

    fig.suptitle(
        "Model 1 post-softmax attention matrices for five random examples\n"
        "Orange lines mark the ANS query row",
        fontsize=16,
    )
    fig.savefig(OUT, dpi=180)

    print(f"seed,{SEED}")
    for idx, (nums, pred) in enumerate(zip(examples, preds), start=1):
        print(f"example_{idx},nums={nums},max={max(nums)},pred={pred}")
    print(f"attention_shape,{tuple(attn.shape)}")
    print(f"wrote,{OUT}")
    print(f"wrote,{EXAMPLES_OUT}")


if __name__ == "__main__":
    main()

