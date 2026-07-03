# First Part: Puzzle 1a

## Objective

Explain how the one-layer, four-head attention-only transformer predicts the
maximum digit from a length-5 list.

## Verified Starting Point

Smoke-test input:

```python
nums = [3, 7, 2, 9, 4]
```

The loaded model predicts `9`, which matches `max(nums)`.

Attention tensor shape for this input:

```text
(batch=1, heads=4, seq=11, seq=11)
```

## Questions To Answer

- Which heads attend from `[ANS]` to number positions?
- Do heads specialize by threshold, rank, position, or token value?
- How much of the answer is available through direct residual stream logits before attention?
- What is each head contributing through `W_V` and `W_O`?
- Can a head or position ablation explain the failure modes cleanly?

## Evidence Checklist

- [ ] Attention heatmaps for representative lists.
- [ ] Direct logit attribution from embedding and head outputs.
- [ ] Head ablations at `[ANS]`.
- [ ] Token-position sweep over all `10^5` inputs or a representative exhaustive subset.
- [ ] Clear description of the final algorithm.

## Notes

Add stable findings here as they become supported by experiments.

