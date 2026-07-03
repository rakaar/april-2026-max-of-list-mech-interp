# April 2026 Max of List Result Book

This is the local result book for the April 2026 Mech Interp Puzzle:
reverse-engineering attention-only transformers trained to predict the maximum
of a length-5 list.

## Current Focus

Start with Puzzle 1a:

- Numbers are single tokens from `0` to `9`.
- Inputs have the form `[BOS] n1 [SEP] n2 [SEP] n3 [SEP] n4 [SEP] n5 [ANS]`.
- The model has one attention layer, four heads, `d_model = 64`, no MLP, and no LayerNorm.
- The target at the `[ANS]` position is the maximum number in the list.

## Local Assets

- Starter notebook: `04_2026/starter_notebook.ipynb`
- Python notebook export: `04_2026/starter_notebook.py`
- Local model definition: `04_2026/model.py`
- Puzzle 1a training script: `04_2026/puzzle1a/train.py`

## Working Rule

Update this book after every useful experiment. Keep scratch exploration in the
notebook, then move stable claims, plots, and evidence into these pages.

