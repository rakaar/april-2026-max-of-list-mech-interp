#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andyrdt/puzzles/blob/main/04_2026/starter_notebook.ipynb)
# 
# # Monthly Algorithmic Challenge — April 2026: Max of List
# 
# *Inspired by Callum McDougall's [ARENA Monthly Algorithmic Challenges](https://learn.arena.education/chapter1_transformer_interp/monthly_algorithmic/).*
# 
# ## Overview
# 
# We've trained two small attention-only transformers to solve the same task: **given a list of 5 numbers, predict the maximum.**
# 
# Both models achieve 100% test accuracy. Your challenge: **reverse-engineer the algorithm each model has learned.**
# 
# ### The two models
# 
# | | Model 1 (easier) | Model 2 (harder) |
# |---|---|---|
# | **Numbers** | 0–9 | 0–99 |
# | **Tokenization** | One token per number | Two digit tokens per number (e.g. 42 → `4`, `2`) |
# | **Layers** | 1 | 2 |
# | **Heads** | 4 | 4 |
# | **`d_model`** | 64 | 64 |
# | **Parameters** | 18,944 | 35,712 |
# | **Vocab** | 14 tokens (0-9 + BOS, SEP, ANS, EOS) | 14 tokens (digits 0-9 + BOS, SEP, ANS, EOS) |
# 
# Both models are trained and evaluated on lists of length 5.
# 
# ### Architecture
# 
# Both models are **attention-only** causal transformers — no MLPs, no LayerNorm. The architecture is:
# 
# ```
# token_embedding + positional_embedding
#     → attention layer(s) with residual connections
#     → linear unembed → logits
# ```
# 
# Positional embeddings are learned (`nn.Embedding`). Each attention layer has `n_heads` independent heads, each computing Q, K, V projections, with a shared W_O output projection. The model returns both logits and per-layer attention patterns.
# 
# ### What constitutes a good solution?
# 
# - **Describe the mechanism** the model uses to find the max.
# - **Provide evidence** via attention pattern visualizations, ablation experiments, activation patching, direct logit attribution, or other relevant techniques.
# - **Present your solution clearly** by utilizing the markdown cells, and presenting clean figures.
# 
# We recommend starting with Model 1 — it's simple enough that you should be able to fully explain every weight matrix.

# ## Setup

# In[ ]:


get_ipython().run_line_magic('pip', 'install -q torch einops nnsight==0.6.3 huggingface_hub matplotlib')


# In[ ]:


import json, importlib
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from nnsight import NNsight
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Download model definition and both sets of weights from HuggingFace
model_py_path = hf_hub_download("andyrdt/04_2026_puzzle_1a", "model.py")
spec = importlib.util.spec_from_file_location("model", model_py_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
AttentionOnlyTransformer = model_module.AttentionOnlyTransformer

# Pre-download both models so later cells don't need network access
config_1_path = hf_hub_download("andyrdt/04_2026_puzzle_1a", "config.json")
weights_1_path = hf_hub_download("andyrdt/04_2026_puzzle_1a", "model.pt")
config_2_path = hf_hub_download("andyrdt/04_2026_puzzle_1b", "config.json")
weights_2_path = hf_hub_download("andyrdt/04_2026_puzzle_1b", "model.pt")

print("Downloaded model definition + weights for both models.")


# ## Helper functions
# 
# Tokenization helpers and a utility to visualize attention patterns. You'll use these throughout.

# In[ ]:


# ── Model 1 vocab (numbers 0-9, each is its own token) ──
NUM_RANGE_1 = 10
BOS_1, SEP_1, ANS_1, EOS_1 = 10, 11, 12, 13
VOCAB_SIZE_1 = 14
TOKEN_NAMES_1 = {10: "BOS", 11: "SEP", 12: "ANS", 13: "EOS"}

def tokenize_1(nums: list[int]) -> list[int]:
    """Tokenize a list of numbers for Model 1.
    Example: [3, 7, 2] -> [BOS, 3, SEP, 7, SEP, 2, ANS]"""
    tokens = [BOS_1]
    for i, n in enumerate(nums):
        tokens.append(n)
        if i < len(nums) - 1:
            tokens.append(SEP_1)
    tokens.append(ANS_1)
    return tokens

def token_labels_1(tokens: list[int]) -> list[str]:
    return [TOKEN_NAMES_1.get(t, str(t)) for t in tokens]


# ── Model 2 vocab (digits 0-9, two per number) ──
BOS_2, SEP_2, ANS_2, EOS_2 = 10, 11, 12, 13
VOCAB_SIZE_2 = 14
TOKEN_NAMES_2 = {10: "BOS", 11: "SEP", 12: "ANS", 13: "EOS"}

def tokenize_2(nums: list[int]) -> list[int]:
    """Tokenize a list of numbers for Model 2.
    Example: [42, 7, 85] -> [BOS, 4, 2, SEP, 0, 7, SEP, 8, 5, ANS]"""
    tokens = [BOS_2]
    for i, n in enumerate(nums):
        tokens.append(n // 10)
        tokens.append(n % 10)
        if i < len(nums) - 1:
            tokens.append(SEP_2)
    tokens.append(ANS_2)
    return tokens

def token_labels_2(tokens: list[int]) -> list[str]:
    return [TOKEN_NAMES_2.get(t, str(t)) for t in tokens]


# ── Attention visualization ──
def _format_attn_ax(ax, attn_matrix, token_labels, title):
    """Format a single attention heatmap axis."""
    ax.imshow(attn_matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=7)
    ax.set_yticks(range(len(token_labels)))
    ax.set_yticklabels(token_labels, fontsize=7)
    ax.set_title(title, fontsize=10)


def plot_attention(attn_patterns, token_labels, title="Attention"):
    """Plot attention heatmaps.

    Args:
        attn_patterns: tensor of shape (n_heads, seq, seq) or list of such tensors
                       (one per layer).
        token_labels: list of strings for tick labels.
        title: plot title.
    """
    if isinstance(attn_patterns, list):
        n_layers = len(attn_patterns)
        n_heads = attn_patterns[0].shape[0]
        fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 3.5 * n_layers))
        if n_layers == 1:
            axes = [axes]
        for layer_idx in range(n_layers):
            attn = attn_patterns[layer_idx].detach().cpu().numpy()
            for h in range(n_heads):
                _format_attn_ax(axes[layer_idx][h], attn[h], token_labels,
                                f"L{layer_idx}H{h}")
    else:
        attn = attn_patterns.detach().cpu().numpy()
        n_heads = attn.shape[0]
        fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 3.5))
        if n_heads == 1:
            axes = [axes]
        for h in range(n_heads):
            _format_attn_ax(axes[h], attn[h], token_labels, f"H{h}")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

print("Helpers loaded.")


# ---
# # Model 1: Max of list (0–9), 1-layer attention-only
# 
# **Task**: Given 5 numbers from 0–9, predict the maximum.
# 
# **Input format**: `[BOS] n1 [SEP] n2 [SEP] n3 [SEP] n4 [SEP] n5 [ANS]`
# 
# **Output**: At the `[ANS]` position, the model should predict the max value. Then it should predict `[EOS]`.
# 
# **Example**: `[BOS] 3 [SEP] 7 [SEP] 2 [SEP] 5 [SEP] 1 [ANS]` → model predicts `7`
# 
# This model has a single attention layer with 4 heads and no MLPs. The entire computation is: embed → one multi-head attention layer (with residual) → unembed.

# In[ ]:


# Load Model 1
config_1 = json.loads(Path(config_1_path).read_text())
raw_model_1 = AttentionOnlyTransformer.from_config(config_1["model"])
raw_model_1.load_state_dict(torch.load(weights_1_path, map_location=device, weights_only=True))
raw_model_1.eval().to(device)
# `NNsight` wraps the model for tracing/interventions, but still exposes the
# underlying module tree and parameters for normal inspection.
model_1 = NNsight(raw_model_1)

print(f"Model 1 config: {config_1['model']}")
print(f"Parameters: {sum(p.numel() for p in model_1.parameters()):,}")


# ### Verifying Model 1 works
# 
# Let's run a few examples and check the model gets them right. We show the full output distribution at the ANS position.

# In[ ]:


examples_1 = [
    [3, 7, 2, 5, 1],
    [0, 0, 0, 0, 0],
    [9, 1, 8, 2, 7],
    [1, 2, 3, 4, 5],
]

fig, axes = plt.subplots(1, len(examples_1), figsize=(4 * len(examples_1), 3))

for idx, nums in enumerate(examples_1):
    tokens = tokenize_1(nums)
    x = torch.tensor([tokens], device=device)

    logits, _ = model_1(x)

    # Softmax over number tokens at the ANS position (last token)
    probs = torch.softmax(logits[0, -1, :NUM_RANGE_1], dim=-1).detach().cpu()
    pred = probs.argmax().item()
    true_max = max(nums)

    ax = axes[idx]
    colors = ["green" if i == true_max else "lightgray" for i in range(NUM_RANGE_1)]
    ax.bar(range(NUM_RANGE_1), probs.numpy(), color=colors)
    ax.set_xticks(range(NUM_RANGE_1))
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Token")
    ax.set_ylabel("P(token)")
    status = "correct" if pred == true_max else "WRONG"
    ax.set_title(f"{nums}\npred={pred}, true={true_max} ({status})")

plt.suptitle("Model 1: output distribution at ANS position", fontsize=13)
plt.tight_layout()
plt.show()


# ### Example: attention patterns for Model 1
# 
# Here's what the 4 attention heads look like on a single input. The ANS row (bottom) is where the model reads from the input to make its prediction.

# In[ ]:


nums = [3, 7, 2, 5, 1]
tokens = tokenize_1(nums)
x = torch.tensor([tokens], device=device)

_, attn_patterns = model_1(x)

# `attn_patterns` is a list with one tensor per layer.
# Each tensor has shape: (batch, n_heads, seq, seq)
attn = attn_patterns[0][0]  # layer 0, batch 0
plot_attention(attn, token_labels_1(tokens), title=f"Model 1: {nums} (max={max(nums)})")


# ### Your turn!
# 
# You now have `model_1` (an `NNsight`-wrapped model) and `raw_model_1` (the underlying plain `nn.Module`).
# 
# The wrapped model still exposes parameters and submodules for direct inspection, while also supporting `NNsight` tracing and interventions.
# 
# Your goal is to try and understand how the model works.
# 
# Good luck!

# ---
# # Model 2: Max of list (0–99), 2-layer attention-only, digit tokenization
# 
# **Task**: Given 5 numbers from 0–99, predict the maximum.
# 
# **Tokenization**: Each number is split into two digit tokens (tens, ones), always zero-padded. So `42` → tokens `4`, `2` and `7` → tokens `0`, `7`.
# 
# **Input format**: `[BOS] d1t d1o [SEP] d2t d2o [SEP] ... d5t d5o [ANS]`
# 
# **Output**: At `[ANS]`, model predicts the tens digit of the max. Then the ones digit. Then `[EOS]`.
# 
# **Example**: `[BOS] 4 2 [SEP] 1 7 [SEP] 8 5 [SEP] 0 3 [SEP] 6 1 [ANS]` → model predicts `8` then `5`
# 
# This model has **2 attention layers** with 4 heads each. The interesting question is how the layers divide the work — a 1-layer model can learn the tens digit (100% accuracy) but plateaus at ~40% for the ones digit.

# In[ ]:


# Load Model 2
config_2 = json.loads(Path(config_2_path).read_text())
raw_model_2 = AttentionOnlyTransformer.from_config(config_2["model"])
raw_model_2.load_state_dict(torch.load(weights_2_path, map_location=device, weights_only=True))
raw_model_2.eval().to(device)
model_2 = NNsight(raw_model_2)

print(f"Model 2 config: {config_2['model']}")
print(f"Parameters: {sum(p.numel() for p in model_2.parameters()):,}")


# ### Verifying Model 2 works
# 
# For Model 2, the output is two tokens: tens digit then ones digit. We feed the input up to `[ANS]`, get the tens prediction, then feed that back to get the ones prediction.

# In[ ]:


examples_2 = [
    [42, 17, 85, 3, 61],
    [99, 0, 50, 25, 75],
    [87, 86, 85, 84, 83],
    [9, 19, 29, 39, 49],
]

for nums in examples_2:
    tokens = tokenize_2(nums)
    x = torch.tensor([tokens], device=device)

    # Get tens digit prediction (at ANS position)
    logits_tens, _ = model_2(x)
    pred_tens = logits_tens[0, -1, :10].argmax().item()

    # Feed tens digit back, get ones digit prediction
    tokens_ext = tokens + [pred_tens]
    x_ext = torch.tensor([tokens_ext], device=device)
    logits_ones, _ = model_2(x_ext)
    pred_ones = logits_ones[0, -1, :10].argmax().item()

    pred_num = pred_tens * 10 + pred_ones
    true_max = max(nums)
    status = "correct" if pred_num == true_max else "WRONG"
    print(f"  {nums} → predicted {pred_num:2d}, true max {true_max:2d}  [{status}]")


# ### Example: attention patterns for Model 2
# 
# With 2 layers, we can see how the model builds up its computation across layers.

# In[ ]:


nums = [42, 17, 85, 3, 61]
tokens = tokenize_2(nums)
x = torch.tensor([tokens], device=device)

_, attn_patterns = model_2(x)

# `attn_patterns` is a list with one tensor per layer.
# Each tensor has shape: (batch, n_heads, seq, seq)
attn_l0 = attn_patterns[0][0]  # layer 0, batch 0
attn_l1 = attn_patterns[1][0]  # layer 1, batch 0

plot_attention([attn_l0, attn_l1], token_labels_2(tokens),
               title=f"Model 2: {nums} (max={max(nums)})")


# ### Your turn!
# 
# You now have `model_2` (an `NNsight`-wrapped model) and `raw_model_2` (the underlying plain `nn.Module`).
# 
# As above, the wrapped model still exposes parameters and submodules for direct inspection, while also supporting `NNsight` tracing and interventions.
# 
# Your goal is to try and understand how the model works.
# 
# Good luck!
