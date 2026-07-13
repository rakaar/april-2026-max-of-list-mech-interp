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

## Attention routing at the `[ANS]` token

### Looking at attention patterns

We only care about the model's response at the `[ANS]` token, which is the
final score vector $F$ in the transformer diagram. In each head, the attention
component that influences $F$ is $a_h$: the vector that says which source
tokens `[ANS]` attends to. Below, we visualize this attention when each digit
$d$ is the maximum number.

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
    Final-row softmax attention for the <code>[ANS]</code> query. Each matrix
    uses the matched input <code>[0, 0, d, 0, 0]</code>, with <code>d</code> at
    the central number position. These are exact representative cases, not
    averages over all inputs with the same maximum.
  </div>
</figure>

### Causal manipulation of the `[ANS]` attention rows

As the maximum number increases, more heads are recruited. The all-head
`[ANS]`-self pattern acts as a baseline that decodes as `0`. A recruited head
changes $a_h$ from `[ANS]` to the maximum-number token, thereby changing
$V_h^a = a_h V_h$ and the residual-stream write $z_h = V_h^a W_O^h$. H3 is
recruited for maxima `2–6`; H2 joins H3 for maxima `7–8`; and H0 joins H2 and
H3 for maximum `9`.

This interpretation makes a causal prediction: changing only $a_h$ should
change the answer, even while every model weight, token embedding, positional
embedding, and all earlier attention rows remain fixed. The prediction holds.

#### H3 steers `[2, 3, 4, 5, 6]`

The unmodified model answers `6`. In each intervention below, $a_0$, $a_1$,
and $a_2$ are forced one-hot to `[ANS]`; only $a_3$ is forced one-hot to a
selected digit position.

| Input | Forced attention rows $a_h$ | Model output |
|---|---|---:|
| `[2, 3, 4, 5, 6]` | $a_0,a_1,a_2 \rightarrow$ `[ANS]`; $a_3 \rightarrow 2$ | **2** |
| `[2, 3, 4, 5, 6]` | $a_0,a_1,a_2 \rightarrow$ `[ANS]`; $a_3 \rightarrow 3$ | **3** |
| `[2, 3, 4, 5, 6]` | $a_0,a_1,a_2 \rightarrow$ `[ANS]`; $a_3 \rightarrow 4$ | **4** |
| `[2, 3, 4, 5, 6]` | $a_0,a_1,a_2 \rightarrow$ `[ANS]`; $a_3 \rightarrow 5$ | **5** |

#### H2 is recruited at `7`

The unmodified model answers `8` for `[4, 5, 6, 7, 8]`. Targets `4–6` use
the lower-number circuit: $a_2$ stays on `[ANS]` while $a_3$ reads the
requested digit. To produce `7`, both $a_2$ and $a_3$ must read the `7` token.

| Input | Forced attention rows $a_h$ | Model output |
|---|---|---:|
| `[4, 5, 6, 7, 8]` | $a_0,a_1,a_2 \rightarrow$ `[ANS]`; $a_3 \rightarrow 4$ | **4** |
| `[4, 5, 6, 7, 8]` | $a_0,a_1,a_2 \rightarrow$ `[ANS]`; $a_3 \rightarrow 5$ | **5** |
| `[4, 5, 6, 7, 8]` | $a_0,a_1,a_2 \rightarrow$ `[ANS]`; $a_3 \rightarrow 6$ | **6** |
| `[4, 5, 6, 7, 8]` | $a_0,a_1 \rightarrow$ `[ANS]`; $a_2,a_3 \rightarrow 7$ | **7** |

## Low-dimensional computation

There are only `10` possible digit answers, while the residual stream has `64`
dimensions. It is therefore reasonable to expect that the answer-writing
computation may use a much lower-dimensional representation.

To test this, we perform PCA on the model's `64 x 64` output matrix $W_O$. If
$Q_k$ contains the top $k$ principal directions, projecting $W_O$ into this
basis reduces it from `64 x 64` to `64 x k`. Equivalently, each head's
`16 x 64` output matrix $W_O^h$ becomes `16 x k`.

We then use the same output-derived basis for the unembedding matrix. This
reduces $W_U$ from `64 x V` to `k x V`. Thus the heads write into a shared
$k$-dimensional space, and the unembedding reads from that same space. The PCA
basis is obtained only from $W_O$; PCA is not fitted separately to $W_U$.

### How many dimensions are needed?

We keep increasing the number of retained principal components and evaluate
the projected computation exhaustively on all `100,000` possible inputs. The
prediction is taken over the full `14`-token vocabulary.

| Output-PCA basis retained | Output-matrix variance captured | Unembedding variance captured | 14-way accuracy |
|---|---:|---:|---:|
| PC1 | 59.23% | 59.25% | 40.952% |
| PC1 + PC2 | 84.06% | 85.04% | 57.758% |
| **PC1 + PC2 + PC3** | **88.34%** | **91.51%** | **100.000%** |

With only three principal components, the projected model reaches `100%`
accuracy. The four head outputs can therefore be added and read by the
unembedding inside a three-dimensional subspace without changing any of the
model's decisions. This shows that the computation needed to solve this task
is low-dimensional.

### Visualizing the three-dimensional computation

Because three dimensions are sufficient, we can plot the ten digit
unembedding vectors and the head outputs in the same space. Both interactives
below use the top three principal directions obtained from $W_O$.

#### Baseline plus recruited corrections

When all heads attend to `[ANS]`, their combined output gives the baseline
answer `0`. The first interactive shows how this baseline is corrected as
heads are recruited to read higher digits. Each arrow shows the change
contributed by a recruited head, and the endpoint shows the resulting summed
head output in the three-dimensional unembedding space.

In the displayed decomposition, the fixed vector $B$ contains the `[ANS]`
writes from H0, H1, and H2, while H3 is shown as the first answer-dependent
arrow. For outputs `7–9`, H2 replaces its `[ANS]` write with a digit write; H0
does the same for output `9`.

[Open the baseline-and-corrections interactive](assets/model1_output_pca_piecewise_interactive.html){ target=_blank .main-results-data-link }

<iframe
  src="../assets/model1_output_pca_piecewise_interactive.html"
  title="Baseline and recruited head corrections in the output-matrix PCA basis"
  style="width: 100%; height: 900px; border: 1px solid #d1d5db;"
  loading="lazy"
  allowfullscreen>
</iframe>

#### Direct output from each head

The second interactive shows the same computation without regrouping it into
a baseline and corrections. Each colored arrow is one head's complete output
$z_h = V_h^a W_O^h$ in the three-dimensional space. The black arrow is the sum
of all four head outputs. The bar chart shows the dot product of this sum with
each vocabulary token's projected unembedding vector.

[Open the direct-head-writes interactive](assets/model1_output_pca_head_contributions_interactive.html){ target=_blank .main-results-data-link }

<iframe
  src="../assets/model1_output_pca_head_contributions_interactive.html"
  title="Four direct head output vectors and their sum in the output-matrix PCA basis"
  style="width: 100%; height: 900px; border: 1px solid #d1d5db;"
  loading="lazy"
  allowfullscreen>
</iframe>
