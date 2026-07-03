#!/usr/bin/env python3
"""Steer concrete examples by changing only the [ANS] attention row."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
JSON_OUT = ROOT / "docs" / "assets" / "model1_counterfactual_attention_steering_examples.json"
NUMBER_POSITIONS = [1, 3, 5, 7, 9]
TOKEN_NAMES = {10: "BOS", 11: "SEP", 12: "ANS", 13: "EOS"}


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


def digit_logits(vec: torch.Tensor, model) -> torch.Tensor:
    return vec @ model.unembed.weight.detach()[:10].T


def format_source(pos: int, tokens: list[int]) -> str:
    token = tokens[pos]
    return f"{TOKEN_NAMES.get(token, str(token))}@{pos}"


def target_position(nums: list[int], target: int) -> int | None:
    for slot, value in enumerate(nums):
        if value == target:
            return NUMBER_POSITIONS[slot]
    return None


def head_value_choices(
    source_values: list[torch.Tensor],
    tokens: list[int],
    nums: list[int],
    target: int,
    alpha_for_one: float | None,
) -> tuple[list[torch.Tensor], dict[str, str]]:
    ans_values = [values[:, 10, :] for values in source_values]
    pos = target_position(nums, target)
    if target != 0 and pos is None:
        raise ValueError(f"target {target} is not present in input {nums}")

    def at_target(head_idx: int) -> torch.Tensor:
        assert pos is not None
        return source_values[head_idx][:, pos, :]

    recipe: dict[str, str] = {}
    if target == 0:
        chosen = ans_values
        recipe = {f"H{i}": "ANS@10" for i in range(4)}
    elif target == 1:
        if alpha_for_one is None:
            raise ValueError("target 1 requires an alpha_for_one mixture")
        h3_mix = (1.0 - alpha_for_one) * ans_values[3] + alpha_for_one * at_target(3)
        chosen = [ans_values[0], ans_values[1], ans_values[2], h3_mix]
        recipe = {
            "H0": "ANS@10",
            "H1": "ANS@10",
            "H2": "ANS@10",
            "H3": f"{1.0 - alpha_for_one:.3f}*ANS@10 + {alpha_for_one:.3f}*{format_source(pos, tokens)}",
        }
    elif 2 <= target <= 6:
        chosen = [ans_values[0], ans_values[1], ans_values[2], at_target(3)]
        recipe = {
            "H0": "ANS@10",
            "H1": "ANS@10",
            "H2": "ANS@10",
            "H3": format_source(pos, tokens),
        }
    elif 7 <= target <= 8:
        chosen = [ans_values[0], ans_values[1], at_target(2), at_target(3)]
        recipe = {
            "H0": "ANS@10",
            "H1": "ANS@10",
            "H2": format_source(pos, tokens),
            "H3": format_source(pos, tokens),
        }
    elif target == 9:
        chosen = [at_target(0), ans_values[1], at_target(2), at_target(3)]
        recipe = {
            "H0": format_source(pos, tokens),
            "H1": "ANS@10",
            "H2": format_source(pos, tokens),
            "H3": format_source(pos, tokens),
        }
    else:
        raise ValueError(f"unsupported target {target}")
    return chosen, recipe


def run_forced_recipe(
    model,
    nums: list[int],
    target: int,
    alpha_for_one: float | None = None,
) -> dict:
    tokens_list = tokenize(nums)
    tokens = torch.tensor([tokens_list], dtype=torch.long)
    seq_len = tokens.shape[1]
    positions = torch.arange(seq_len).unsqueeze(0)

    with torch.no_grad():
        resid = model.tok_embed(tokens) + model.pos_embed(positions)
        ans_resid = resid[:, 10, :]
        layer = model.layers[0]
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)

        actual_logits, _ = model(tokens)
        actual_digit_logits = actual_logits[:, 10, :10]

        source_values = []
        actual_top_sources = {}
        for head_idx, head in enumerate(layer.heads):
            _, attn = head(resid, mask)
            attn_row = attn[0, 10, :]
            top_pos = int(attn_row.argmax())
            actual_top_sources[f"H{head_idx}"] = {
                "source": format_source(top_pos, tokens_list),
                "mass": float(attn_row[top_pos]),
            }
            source_values.append(resid @ head.W_V.weight.detach().T)

        chosen, recipe = head_value_choices(source_values, tokens_list, nums, target, alpha_for_one)
        head_sum = layer.W_O(torch.cat(chosen, dim=-1))
        final_vec = ans_resid + head_sum
        forced_logits = digit_logits(final_vec, model)[0]
        pred = int(forced_logits.argmax())
        top2 = torch.topk(forced_logits, k=2)
        margin = float(top2.values[0] - top2.values[1])

    return {
        "target": target,
        "alpha_for_one": alpha_for_one,
        "prediction": pred,
        "top_logit": float(top2.values[0]),
        "runner_up_logit": float(top2.values[1]),
        "top_minus_runner_up": margin,
        "digit_logits": [float(v) for v in forced_logits],
        "recipe": recipe,
        "actual_prediction": int(actual_digit_logits.argmax()),
        "actual_top_sources": actual_top_sources,
    }


def alpha_scan_for_one(model, nums: list[int]) -> dict | None:
    if 1 not in nums:
        return None
    rows = []
    best = None
    in_interval = False
    start_alpha = None
    intervals = []
    for step in range(1001):
        alpha = step / 1000.0
        row = run_forced_recipe(model, nums, 1, alpha_for_one=alpha)
        logits = torch.tensor(row["digit_logits"])
        target_margin = float(logits[1] - torch.cat([logits[:1], logits[2:]]).max())
        row["target1_margin_vs_best_other"] = target_margin
        rows.append(row)
        if best is None or target_margin > best["target1_margin_vs_best_other"]:
            best = row
        if row["prediction"] == 1 and not in_interval:
            start_alpha = alpha
            in_interval = True
        if row["prediction"] != 1 and in_interval:
            intervals.append([start_alpha, (step - 1) / 1000.0])
            in_interval = False
    if in_interval:
        intervals.append([start_alpha, 1.0])
    assert best is not None
    return {
        "best_alpha": best["alpha_for_one"],
        "best_prediction": best["prediction"],
        "best_target1_margin_vs_best_other": best["target1_margin_vs_best_other"],
        "prediction_1_alpha_intervals": intervals,
        "sample_predictions": {
            f"{alpha:.1f}": run_forced_recipe(model, nums, 1, alpha_for_one=alpha)["prediction"]
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]
        },
    }


def run_example(model, nums: list[int], targets: list[int]) -> dict:
    actual = run_forced_recipe(model, nums, 0)
    alpha_info = alpha_scan_for_one(model, nums)
    rows = []
    for target in targets:
        if target == 1:
            if alpha_info is None:
                rows.append({"target": target, "skipped": "target 1 not present"})
                continue
            alpha = float(alpha_info["best_alpha"])
            rows.append(run_forced_recipe(model, nums, target, alpha_for_one=alpha))
        elif target != 0 and target not in nums:
            rows.append({"target": target, "skipped": f"target {target} not present"})
        else:
            rows.append(run_forced_recipe(model, nums, target))

    return {
        "nums": nums,
        "true_max": max(nums),
        "actual_prediction": actual["actual_prediction"],
        "actual_top_sources": actual["actual_top_sources"],
        "target1_alpha_scan": alpha_info,
        "forced_targets": rows,
    }


def main() -> None:
    torch.manual_seed(0)
    model = load_model()
    examples = [
        {"nums": [1, 2, 3, 4, 5], "targets": [0, 1, 2, 3, 4, 5]},
        {"nums": [5, 6, 7, 8, 9], "targets": [0, 5, 6, 7, 8, 9]},
        {"nums": [0, 1, 2, 7, 9], "targets": [0, 1, 2, 7, 9]},
    ]
    result = {
        "description": (
            "Counterfactual steering by replacing only the [ANS] attention row for each head. "
            "Allowed source choices are [ANS] self or a present source digit. Target 1 uses a "
            "searched H3 mixture between [ANS] and source digit 1."
        ),
        "examples": [run_example(model, item["nums"], item["targets"]) for item in examples],
    }
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    for example in result["examples"]:
        print(f"input,{example['nums']},true_max,{example['true_max']},actual_pred,{example['actual_prediction']}")
        if example["target1_alpha_scan"] is not None:
            scan = example["target1_alpha_scan"]
            print(
                "target1_alpha,"
                f"best:{scan['best_alpha']:.3f},"
                f"intervals:{scan['prediction_1_alpha_intervals']},"
                f"samples:{scan['sample_predictions']}"
            )
        print("target,pred,margin,recipe")
        for row in example["forced_targets"]:
            if "skipped" in row:
                print(f"{row['target']},SKIP,{row['skipped']}")
                continue
            recipe = "; ".join(f"{head}->{source}" for head, source in row["recipe"].items())
            print(f"{row['target']},{row['prediction']},{row['top_minus_runner_up']:.6f},{recipe}")
        print()
    print(f"wrote,{JSON_OUT}")


if __name__ == "__main__":
    main()
