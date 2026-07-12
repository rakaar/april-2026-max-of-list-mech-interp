---
hide:
  - navigation
toc_depth: 3
---

<div data-main-results-page hidden></div>

# Main results

## Model and notation

### Transformer architecture

<div
  class="architecture-diagram"
  data-transformer-architecture
  role="img"
  aria-label="One-layer attention-only transformer architecture with matrix notation"
>
  <div class="architecture-loading">Loading transformer architecture...</div>
</div>

<div class="notation-grid" markdown>

<div class="notation-matrices" markdown>

### Matrices

| Matrix | Shape | Symbol |
|---|---:|---:|
| Embedding matrix | `14 x 64` | $W_E$ |
| Unembedding matrix | `64 x 14` | $W_U$ |
| Query matrix of head "h" | `64 x 16` | $W_Q^h$ |
| Key matrix of head "h" | `64 x 16` | $W_K^h$ |
| Value matrix of head "h" | `64 x 16` | $W_V^h$ |
| $W_O$ of each head | `16 x 64` | $W_O^h$ |

It is generally considered to be one large matrix while computing (`64 x 64`).
But to interpret its easy to think that each head has its own output matrix.

</div>

<div class="notation-vectors" markdown>

### Value vector

Value vector after doing weighted attention of head h (`N x 16`, but last row
would suffice to explain so considered `1 x 16`): $V_h$, which is basically
(Attention x ($W_V$ x residual stream)).

### Sum of head outputs

Sum of all heads output just before multiplying with unembdding matrix (`N x
64`, but last row suffices so `1 x 64`):

$$
\Sigma_h W_O^h V_h
$$

!!! note "Notation check"
    The draft above is preserved. With the displayed row-vector shapes,
    $V_h$ is `1 x 16` and $W_O^h$ is `16 x 64`, so the dimensionally
    consistent product is $V_h W_O^h$. The summed final-row write is therefore
    $\sum_h V_h W_O^h$. PyTorch stores linear-layer weights transposed relative
    to this mathematical row-vector convention.

</div>

</div>

## Attention routing at the `[ANS]` token

### Looking at attention patterns in each head at ANS token

The final response we care about is the prediction of the model at last token
(ANS token). So its sufficient to look at last row of the prediction logits
which is of size `N x Vocab Size`. We only need `-1 x Vocab size`. If we trace
back down, in each head, all that matters for the computation is the last row
of the attention matrix. The last row of the attention matrix checks what
tokens the ANS token attends to?

Below is the plot for

$$
W_Q^h [\text{ANS token in residual stream}] \cdot
W_K^h [\text{Embedding vector of n}]
$$

> **Display specification**
>
> Show every maximum from `0` through `9` separately. In each case, show a
> colored `4 x 11` matrix: four attention heads by `11` source tokens, with token
> identities at the top. Matrices are laid out in rows as:
> `0`, `1`, `2–6`, `7–8`, and `9` so that similar regimes are easy to compare.

<figure class="main-results-plot attention-explorer">
  <div
    class="attention-regime-grid"
    data-attention-source="../assets/main_results_ans_attention_regimes.json"
    aria-live="polite"
  >
    <div class="attention-loading">Loading exact attention matrices...</div>
  </div>
  <noscript>
    <picture>
      <source
        media="(max-width: 760px)"
        srcset="../assets/main_results_ans_attention_by_max_mobile.png"
      >
      <img
        src="../assets/main_results_ans_attention_by_max.png"
        alt="Ten heatmaps showing the exact ANS attention row for four heads across all eleven source tokens"
      >
    </picture>
  </noscript>
  <div class="main-results-figure-caption">
    Actual final-row softmax attention for the <code>[ANS]</code> query. Each
    matrix has four head rows and eleven source-token columns. Matrices are shown in grouped rows (`0`, `1`, `2–6`, `7–8`, `9`) so similar regimes can be compared side by side. All ten maxima are shown separately; no attention matrices are averaged.
  </div>
</figure>

[Open the exact plotted values](assets/main_results_ans_attention_regimes.json){ .main-results-data-link }

- $W_Q$ ANS x $W_K$ max number vs $W_Q$ ANS x $W_K$ ANS in each head

!!! info "How to read this diagnostic"
    These are the model's actual attention distributions after softmax over all
    `11` source positions. The coral outline marks the largest entry in each
    head row. Inputs use the matched form `[0, 0, m, 0, 0]`, with the unique
    nonzero maximum at source position `5`.

    Max `1` is the important soft case: H3 gives approximately `62%` to
    `[ANS]` and `38%` to the `1` token. From max `2` onward, the recruited
    heads place nearly all their attention on the maximum token.

### Causal manipulation of the `[ANS]` attention rows

As the maximum number increases, more heads are recruited. The all-head
`[ANS]`-self pattern acts as a baseline that decodes as `0`. A recruited head
changes its final attention row from `[ANS]` to the maximum-number token,
thereby adding a number-dependent correction through its value and output
matrices. H3 is recruited for maxima `2–6`; H2 joins H3 for maxima `7–8`; and
H0 joins H2 and H3 for maximum `9`.

This interpretation makes a causal prediction: changing only the final
`[ANS]` attention rows should change the answer, even while every model weight,
token embedding, positional embedding, and all earlier attention rows remain
fixed. The prediction holds.

#### H3 steers `[2, 3, 4, 5, 6]`

The unmodified model answers `6`. In each intervention below, H0, H1, and H2
are forced one-hot to `[ANS]`; only H3 is forced one-hot to a selected digit
position.

| Input | Forced final `[ANS]` attention rows | Model output |
|---|---|---:|
| `[2, 3, 4, 5, 6]` | H0/H1/H2 $\rightarrow$ `[ANS]`; H3 $\rightarrow$ `2` | **2** |
| `[2, 3, 4, 5, 6]` | H0/H1/H2 $\rightarrow$ `[ANS]`; H3 $\rightarrow$ `3` | **3** |
| `[2, 3, 4, 5, 6]` | H0/H1/H2 $\rightarrow$ `[ANS]`; H3 $\rightarrow$ `4` | **4** |
| `[2, 3, 4, 5, 6]` | H0/H1/H2 $\rightarrow$ `[ANS]`; H3 $\rightarrow$ `5` | **5** |

#### H2 is recruited at `7`

The unmodified model answers `8` for `[4, 5, 6, 7, 8]`. Targets `4–6` use
the lower-number circuit: H2 stays on `[ANS]` while H3 reads the requested
digit. To produce `7`, both H2 and H3 must read the `7` token.

| Input | Forced final `[ANS]` attention rows | Model output |
|---|---|---:|
| `[4, 5, 6, 7, 8]` | H0/H1/H2 $\rightarrow$ `[ANS]`; H3 $\rightarrow$ `4` | **4** |
| `[4, 5, 6, 7, 8]` | H0/H1/H2 $\rightarrow$ `[ANS]`; H3 $\rightarrow$ `5` | **5** |
| `[4, 5, 6, 7, 8]` | H0/H1/H2 $\rightarrow$ `[ANS]`; H3 $\rightarrow$ `6` | **6** |
| `[4, 5, 6, 7, 8]` | H0/H1 $\rightarrow$ `[ANS]`; H2/H3 $\rightarrow$ `7` | **7** |

!!! success "Causal result"
    The true maxima are `6` and `8`, but replacing only the `[ANS]` query row
    of the attention matrices makes the frozen model output the selected
    non-maximum digit. The attention pattern therefore does not merely
    correlate with the answer: the head-specific source choices causally
    determine it.

The intervention is implemented by replacing each selected `1 x 11`
post-softmax attention row with a one-hot row, then applying the model's
unchanged $W_V^h$, $W_O^h$, residual addition, and $W_U$. The special maximum
`1` regime is excluded here because H3 uses a soft `[ANS]`/`1` mixture rather
than a one-hot row.

Reproducible analysis:
`scripts/analysis/model1_counterfactual_attention_steering_examples.py`.
[Open the exact logits and interventions](assets/model1_counterfactual_attention_steering_examples.json){ .main-results-data-link }

## Low-dimensional computation

Except for the soft-attention case at maximum `1`, the successful one-hot
interventions let us treat each head's final value as a choice between its
source value at `[ANS]`, $V_h[\mathrm{ANS}]$, and its source value at the
selected number, $V_h[n]$. Here $V_h[n]$ is shorthand for the value of the
complete residual vector at that source position, including both token and
positional embedding.

In row-vector notation, each selected `1 x 16` value is mapped into the
`64`-dimensional residual stream by its head's `16 x 64` output matrix:

$$
z_h = V_h W_O^h \in \mathbb{R}^{1 \times 64},
\qquad
z = \sum_{h=0}^{3} z_h.
$$

The head writes are added and scored against the unembedding matrix:

$$
\ell = z W_U \in \mathbb{R}^{1 \times |\mathcal{V}|},
\qquad
\widehat{y} = \operatorname*{argmax}_t \ell_t.
$$

For this model, the attention-head sum $z$ by itself already gives `100%`
accuracy over all `100,000` possible five-digit inputs. The experiment below
therefore isolates this sufficient attention-write computation; it does not
add the original `[ANS]` residual before unembedding.

### Output-matrix PCA

Let the four output maps be stacked into the mathematical `64 x 64` output
matrix:

$$
O_{\mathrm{all}} =
\begin{bmatrix}
W_O^0 \\
W_O^1 \\
W_O^2 \\
W_O^3
\end{bmatrix}.
$$

PCA is fitted after centering the `64` rows of $O_{\mathrm{all}}$. If $Q_k$
contains its top $k$ principal directions (`64 x k`), each head's output map
and the unembedding can be expressed in the same reduced basis:

$$
\widetilde{W}_O^h = W_O^h Q_k
    \in \mathbb{R}^{16 \times k},
\qquad
\widetilde{W}_U = Q_k^\top W_{U,c}
    \in \mathbb{R}^{k \times |\mathcal{V}|}.
$$

The complete reduced computation is then

$$
z_k = \sum_h V_h \widetilde{W}_O^h
    \in \mathbb{R}^{1 \times k},
\qquad
\ell_k = z_k \widetilde{W}_U.
$$

$W_{U,c}$ is centered over vocabulary items. This subtracts the same scalar
from every candidate logit and therefore cannot change the argmax. Although
centering is used to fit PCA, no output-matrix mean is subtracted from the
actual head writes.

### Accuracy as dimensions are added

The projected computation was evaluated exhaustively on all `100,000` inputs
using the full `14`-token vocabulary. Every column below uses the same basis
$Q_k$, obtained only from PCA of the centered output matrix. The output column
reports how much centered output-matrix variance this basis captures. The
unembedding column reports how much centered `14 x 64` unembedding variance is
captured after projecting it onto that same output-derived basis; PCA is not
fitted to the unembedding matrix.

| Output-PCA basis retained | Output-matrix variance captured | Unembedding variance captured | 14-way accuracy |
|---|---:|---:|---:|
| PC1 | 59.23% | 59.25% | 40.952% |
| PC1 + PC2 | 84.06% | 85.04% | 57.758% |
| **PC1 + PC2 + PC3** | **88.34%** | **91.51%** | **100.000%** |

PC3 contributes only `4.28%` additional output-matrix variance, but raises
exhaustive accuracy from `57.758%` to `100%`.

!!! success "Three dimensions are sufficient"
    Each head's learned `16 x 64` output map can be replaced, for this task, by
    the derived `16 x 3` map $W_O^h Q_3$. The four `1 x 3` head writes are
    summed and scored against the unembedding projected into the same basis.
    This reduced computation preserves every one of the model's `100,000`
    max-of-five decisions.

### Interactive three-dimensional computation

Because three dimensions suffice, the answer-writing computation can be shown
directly. Both panels use the same $Q_3$ basis obtained from PCA of the output
matrix.

#### Baseline plus recruited corrections

The first panel presents the computation as a fixed baseline followed by
answer-dependent corrections. The baseline is
$B=H0([\mathrm{ANS}])+H1([\mathrm{ANS}])+H2([\mathrm{ANS}])$. H3 supplies the
first answer-dependent write. For outputs `7–9`, H2's `[ANS]` write is replaced
by its number write; for output `9`, H0 is replaced in the same way. Thus the
H2 and H0 arrows are replacement differences such as
$H2(n)-H2([\mathrm{ANS}])$, avoiding double-counting the self write already in
$B$.

[Open the baseline-and-corrections interactive](assets/model1_output_pca_piecewise_interactive.html){ target=_blank .main-results-data-link }

<iframe
  src="../assets/model1_output_pca_piecewise_interactive.html"
  title="Baseline and recruited head corrections in the output-matrix PCA basis"
  style="width: 100%; height: 900px; border: 1px solid #d1d5db;"
  loading="lazy"
  allowfullscreen>
</iframe>

Exact values:
[model1_output_pca_piecewise_interactive.json](assets/model1_output_pca_piecewise_interactive.json).
Source: `scripts/analysis/model1_output_pca_piecewise_interactive.py`.

#### Direct output from each head

The second panel regroups the same endpoint as four direct head writes rather
than a baseline and corrections. Each colored arrow starts at the origin and
is the projected vector $V_hW_O^h$. The black arrow is their sum
$z=\sum_h V_hW_O^h$. Select an output `0–9` to see which source each head reads
and how the resulting sum scores all `14` vocabulary tokens.

For output `1`, H3 uses its measured soft `[ANS]`/`1` attention row. Every
other endpoint uses the verified one-hot attention recipe. For all ten
requested outputs, both the `3d` and full `64d` head sums predict the requested
token.

[Open the direct-head-writes interactive](assets/model1_output_pca_head_contributions_interactive.html){ target=_blank .main-results-data-link }

<iframe
  src="../assets/model1_output_pca_head_contributions_interactive.html"
  title="Four direct head output vectors and their sum in the output-matrix PCA basis"
  style="width: 100%; height: 900px; border: 1px solid #d1d5db;"
  loading="lazy"
  allowfullscreen>
</iframe>

Exact values:
[model1_output_pca_head_contributions_interactive.json](assets/model1_output_pca_head_contributions_interactive.json).
Source:
`scripts/analysis/model1_output_pca_head_contributions_interactive.py`.

The same three output-derived directions capture `88.34%` of the centered
output matrix's variance and about `91.5%` of the centered unembedding
variance. This supports a shared low-dimensional read/write subspace: the
heads write answer-relevant information into directions that the unembedding
also reads strongly.

This is a sufficiency result, not a matrix-rank claim. The centered output
matrix has rank `63`, and `11.66%` of its variance lies outside the three-PC
subspace. Those discarded directions may change logit values, but they are not
needed to preserve the argmax on this complete input space. The two interactive
panels above decompose the head-specific `[ANS]` baseline and recruited
corrections inside this `3d` space.

Reproducible analysis:
`scripts/analysis/model1_output_pca_readout_accuracy.py`.
[Open the exact PCA, variance, and accuracy values](assets/model1_output_pca_readout_accuracy.json){ .main-results-data-link }

### Unembedding-matrix PCA works too

The construction also works in the other direction. Instead of obtaining the
low-dimensional basis from the output matrix, PCA can be fitted to the
centered full `14 x 64` unembedding matrix. The resulting `64 x k` basis is
then used to reduce every head's `16 x 64` output map to `16 x k`. The four
heads write and sum directly in `k` dimensions before the full `14`-token
unembedding readout.

This version was also evaluated exhaustively over all `100,000` inputs. Every
column below uses the basis obtained only from full-vocabulary unembedding
PCA. The output-matrix variance column measures how much variance that same
unembedding-derived basis captures after being applied to the centered output
matrix.

| Full-unembedding PCA basis retained | Unembedding variance captured | Output-matrix variance captured | 14-way accuracy |
|---|---:|---:|---:|
| PC1 | 62.01% | 56.45% | 40.952% |
| PC1 + PC2 | 87.37% | 82.01% | 86.318% |
| **PC1 + PC2 + PC3** | **94.04%** | **86.23%** | **100.000%** |

!!! success "Either matrix supplies a sufficient three-dimensional basis"
    Using the top three full-unembedding PCs, each head's `16 x 64` output map
    can be replaced by a derived `16 x 3` map. The resulting three-dimensional
    computation predicts the correct token for all `100,000` inputs and never
    predicts `[BOS]`, `[SEP]`, `[ANS]`, or `[EOS]`.

### Why both bases work

The leading output and unembedding directions are close in residual-stream
space. The [July 12 PC-alignment experiment](2026-07-12.md#model-1-are-the-w_o-and-w_u-top-three-pc-subspaces-the-same)
first showed this using the ten digit-unembedding rows. Repeating the same
calculation with the full `14`-token unembedding basis used in the table above
gives the following exact `64d` cosine matrix:

| | Output PC1 | Output PC2 | Output PC3 |
|---|---:|---:|---:|
| **Full-$W_U$ PC1** | **0.9689** | 0.1840 | 0.0302 |
| **Full-$W_U$ PC2** | -0.2011 | **0.9655** | 0.0703 |
| **Full-$W_U$ PC3** | -0.0267 | -0.0763 | **0.9676** |

| Comparison | PC1 | PC2 | PC3 |
|---|---:|---:|---:|
| Same-index PC cosine | 0.9689 | 0.9655 | 0.9676 |
| Principal angle between the two top-three subspaces | 5.24 degrees | 11.03 degrees | 14.34 degrees |

The bases are strongly aligned but not identical. This overlap explains why
either set of PCs captures most of the other matrix's variance and preserves
the same low-dimensional read/write computation. Alignment alone does not
guarantee perfect accuracy: the remaining requirement is that discarding the
other directions never moves an input across a competing token's
dot-product decision boundary. Exhaustive evaluation confirms that condition
for this task at `k = 3`.

Reproducible analysis:
`scripts/analysis/model1_unembedding_pca_readout_accuracy.py`.
[Open the exact full-unembedding PCA, alignment, variance, and accuracy values](assets/model1_unembedding_pca_readout_accuracy.json){ .main-results-data-link }
