# Attention Head `[ANS]` Row Manipulation

This note records a causal steering test for Model 1 of the April 2026
Baulab max-of-list puzzle.

The question was whether the current attention-level explanation is strong
enough to steer the model to output a requested number, not necessarily the
true maximum, by changing only the last row of each attention matrix: the
`[ANS]` query row.

## Setup

Input format:

```text
[BOS] n0 [SEP] n1 [SEP] n2 [SEP] n3 [SEP] n4 [ANS]
```

The number-token positions are:

```text
1, 3, 5, 7, 9
```

The `[ANS]` position is:

```text
10
```

For each head `h`, the `[ANS]` output is:

```text
head_h = attention_h[ANS, :] @ V_h @ W_O_h
```

Shape details:

```text
attention_h[ANS, :]     1 x 11
V_h source values       11 x 16
after attention         1 x 16
after W_O_h             1 x 64
```

The four head outputs are summed and read through the digit unembedding:

```text
head_sum = H0 + H1 + H2 + H3          # 1 x 64
logits = (ans_resid + head_sum) @ W_U[0:10].T
prediction = argmax(logits)
```

In this experiment, the only manipulated object is the `[ANS]` attention row
for each head. The source choices are restricted to:

- `[ANS]` itself at position `10`;
- a number token that is present in the input.

Equivalently, the experiment replaces the attention-weighted value read at the
last row with a chosen source value, then applies the same learned `W_O_h`.

## Attention Recipe

The current abstraction is:

| Requested output | H0 `[ANS]` row | H1 `[ANS]` row | H2 `[ANS]` row | H3 `[ANS]` row |
|---:|---|---|---|---|
| 0 | `[ANS]` | `[ANS]` | `[ANS]` | `[ANS]` |
| 1 | `[ANS]` | `[ANS]` | `[ANS]` | soft mixture of `[ANS]` and token `1` |
| 2..6 | `[ANS]` | `[ANS]` | `[ANS]` | requested number |
| 7..8 | `[ANS]` | `[ANS]` | requested number | requested number |
| 9 | requested number | `[ANS]` | requested number | requested number |

The special case is output `1`. One-hot H3 attention to token `1` overshoots
and predicts `2`; H3 needs a soft interpolation between `[ANS]` and token `1`.

## Counterfactual Steering Results

Repro script:

```bash
.venv/bin/python scripts/analysis/model1_counterfactual_attention_steering_examples.py
```

Exact output:

```text
docs/assets/model1_counterfactual_attention_steering_examples.json
```

### Example: `[1, 2, 3, 4, 5]`

Baseline prediction is `5`.

After forcing the `[ANS]` rows:

| Requested output | Model output | Margin over runner-up | Forced recipe |
|---:|---:|---:|---|
| 0 | 0 | +12.539337 | H0/H1/H2/H3 -> `[ANS]` |
| 1 | 1 | +22.153854 | H0/H1/H2 -> `[ANS]`; H3 = `0.429 * [ANS] + 0.571 * 1@1` |
| 2 | 2 | +11.884033 | H0/H1/H2 -> `[ANS]`; H3 -> `2@3` |
| 3 | 3 | +12.988159 | H0/H1/H2 -> `[ANS]`; H3 -> `3@5` |
| 4 | 4 | +14.046257 | H0/H1/H2 -> `[ANS]`; H3 -> `4@7` |
| 5 | 5 | +14.939819 | H0/H1/H2 -> `[ANS]`; H3 -> `5@9` |

For target `1`, an alpha scan over:

```text
H3 = (1 - alpha) * [ANS] + alpha * token 1
```

found that prediction `1` occurs for approximately:

```text
alpha in [0.207, 0.873]
```

The best target-`1` margin occurred at:

```text
alpha = 0.571
```

Sample target-`1` outputs:

| alpha | output |
|---:|---:|
| 0.0 | 0 |
| 0.25 | 1 |
| 0.5 | 1 |
| 0.75 | 1 |
| 1.0 | 2 |

### Example: `[5, 6, 7, 8, 9]`

Baseline prediction is `9`.

After forcing the `[ANS]` rows:

| Requested output | Model output | Margin over runner-up | Forced recipe |
|---:|---:|---:|---|
| 0 | 0 | +12.539337 | H0/H1/H2/H3 -> `[ANS]` |
| 5 | 5 | +14.909561 | H0/H1/H2 -> `[ANS]`; H3 -> `5@1` |
| 6 | 6 | +16.117569 | H0/H1/H2 -> `[ANS]`; H3 -> `6@3` |
| 7 | 7 | +18.525780 | H0/H1 -> `[ANS]`; H2/H3 -> `7@5` |
| 8 | 8 | +9.037857 | H0/H1 -> `[ANS]`; H2/H3 -> `8@7` |
| 9 | 9 | +19.581543 | H1 -> `[ANS]`; H0/H2/H3 -> `9@9` |

### Example: `[0, 1, 2, 7, 9]`

Baseline prediction is `9`.

After forcing the `[ANS]` rows:

| Requested output | Model output | Margin over runner-up | Forced recipe |
|---:|---:|---:|---|
| 0 | 0 | +12.539337 | H0/H1/H2/H3 -> `[ANS]` |
| 1 | 1 | +22.157990 | H0/H1/H2 -> `[ANS]`; H3 = `0.429 * [ANS] + 0.571 * 1@3` |
| 2 | 2 | +11.851089 | H0/H1/H2 -> `[ANS]`; H3 -> `2@5` |
| 7 | 7 | +18.641426 | H0/H1 -> `[ANS]`; H2/H3 -> `7@7` |
| 9 | 9 | +19.581543 | H1 -> `[ANS]`; H0/H2/H3 -> `9@9` |

## Interpretation

This is a strong causal check of the attention-row mechanism.

The model can be steered to output a non-maximum number by changing only the
last row of the attention matrices, as long as the forced rows follow the
same head-specific recipe discovered from the real model:

```text
0/1:    [ANS] baseline and H3 soft interpolation
2..6:   H3 reads the requested number
7/8:    H2 and H3 read the requested number
9:      H0, H2, and H3 read the requested number
```

For `[1, 2, 3, 4, 5]`, this intervention can make the model output every
requested digit from `0` through `5`, even though the true maximum is `5`.

For `[5, 6, 7, 8, 9]`, the same intervention can make the model output
`0, 5, 6, 7, 8, 9`.

This supports the view that the `[ANS]` row is the causal control surface for
answer selection. The heads are not merely correlated with the answer; their
source choices determine which 64d write is added at `[ANS]`, and that write
lands in the corresponding digit region after unembedding.

## Related Low-Dimensional Result

The attention-row recipes also land in the 2D PCA decision plane discovered
from the actual head-sum outputs:

```text
scripts/analysis/model1_head_sum_pca_lowdim.py
scripts/analysis/model1_piecewise_scheme_pca_projection.py
docs/assets/model1_head_sum_pca_lowdim.png
docs/assets/model1_piecewise_scheme_pca_projection.png
```

The top two PCs of the 64d head-sum output preserve `100%` digit accuracy over
all `100000` inputs. The piecewise attention recipes also preserve `100%`
accuracy after reconstruction from those same top two PCs.

So the causal story can be summarized as:

```text
force [ANS] attention rows
    -> choose per-head source values
    -> write four 64d vectors through W_O
    -> summed vector lands in a 2D answer plane
    -> unembedding decision region gives the requested digit
```
