# April 2026 Max-of-List Mechanistic Interpretation

Working analysis for the April 2026 Baulab mech-interp puzzle:

https://puzzles.baulab.info/april-2026.html

The focus here is Model 1 from the first part of the challenge: an
attention-only transformer that returns the maximum digit from a list of five
digits.

## Result Book

- [Main results](https://rakaar.github.io/april-2026-max-of-list-mech-interp/main-results/)
- [Research log](https://rakaar.github.io/april-2026-max-of-list-mech-interp/experiments/)

## Summary of Main Results

The model implements a piecewise attention circuit at the final `[ANS]`
position. H3 reads maxima `2-6`; H2 joins H3 for `7-8`; and H0 joins them for
`9`. H1 remains a nearly constant `[ANS]`-reading head, while maxima `0` and `1`
use the `[ANS]` baseline and a soft H3 mixture respectively.

Replacing only the final attention row of each head with this circuit preserves
the model's predictions on all `100,000` possible five-digit inputs. The four
head writes are summed in the residual stream, and the complete computation can
be reduced to three dimensions using PCs of either the stacked output matrix or
the full-vocabulary unembedding matrix without losing accuracy.

The readout is based on dot products, not angle alone. A retrained model whose
unembedding rows are constrained to unit norm has exact angular decoding in the
full `64`-dimensional space. Its top-three projection still has unequal
unembedding norms: dot-product accuracy is `100%`, while cosine-only accuracy
is `95.317%`.

## Key Artifacts

- `attention_head_ANS_row_manipulation.md`: standalone causal steering result.
  It shows that changing only the `[ANS]` row of each attention head can steer
  concrete examples to output requested non-maximum numbers.
- `docs/2026-07-02.md`: result-book page with the low-dimensional head-sum
  PCA result and related analyses.
- `docs/assets/model1_head_sum_pca_lowdim.png`: PCA of the 64d summed head
  output at `[ANS]`.
- `docs/assets/model1_piecewise_scheme_pca_projection.png`: piecewise
  attention recipes projected into the same 2D PCA answer plane.

## Reproduction

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-docs.txt
```

Run the main steering experiment:

```bash
python scripts/analysis/model1_counterfactual_attention_steering_examples.py
```

Run the low-dimensional PCA analyses:

```bash
python scripts/analysis/model1_head_sum_pca_lowdim.py
python scripts/analysis/model1_piecewise_scheme_pca_projection.py
```

Build the result book:

```bash
mkdocs build --strict
```

Preview locally:

```bash
mkdocs serve --dev-addr 127.0.0.1:8000
```

## Layout

- `04_2026/`: upstream starter notebook, Python export, and model file.
- `scripts/analysis/`: reproducible analysis scripts.
- `docs/`: MkDocs result book.
- `docs/assets/`: generated figures and JSON summaries used by the result
  book.
- `requirements.txt`: model-analysis dependencies.
- `requirements-docs.txt`: result-book dependencies.

The repository intentionally ignores local virtualenvs, MkDocs build output,
runtime logs, tunnel state, and local Codex session files.
