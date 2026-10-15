# Experiment: Dictionary Recovery in a One-Sparse Autoencoder

## 1. Research question

**Can a standard TopK sparse autoencoder (SAE), trained by gradient descent, recover the original token embeddings of a pretrained language model?**

We deliberately choose a setting in which an exact-reconstruction dictionary is mathematically identifiable, allowing us to investigate the effectiveness of SAE training rather than the existence of an appropriate representation.

## 2. Experimental setup

We use the token embedding matrix of **GPT-2 Small**:

\[
E\in\mathbb R^{V\times d},\qquad V=50{,}257,\quad d=768.
\]

Each row \(e_i\in\mathbb R^d\) is one training example. We train a TopK SAE with:

- **Dictionary size:** \(m=V=50{,}257\).
- **Sparsity:** \(k=1\), so at most one feature activates per input.
- **Decoder:** Unit-norm columns, enforced throughout training.
- **Decoder bias:** Fixed to zero.
- **Initialization:** Random, with no information about the original embedding directions.
- **Loss:** Mean squared reconstruction error, without an additional sparsity penalty.

## 3. Mathematical expectation

There is an exact solution in which the decoder feature vectors are the normalized token embeddings:

\[
v_i=\frac{e_i}{\|e_i\|_2}.
\]

Provided the embeddings are nonzero and no two lie on the same positive ray, any **zero-reconstruction-error** solution must recover these directions up to permutation (assuming nonnegative one-sparse codes and zero decoder bias).

**The central empirical question is whether gradient-based training actually finds this known global optimum.**

## 4. Measurements

| Metric | Purpose |
|---|---|
| Reconstruction MSE | Does training approach zero error? |
| Dictionary recovery | How closely do learned features match token embeddings under one-to-one assignment? |
| Dead features | How many decoder vectors are never selected? |
| Stability across seeds | How reliably does training recover the same dictionary? |

We compare randomly initialized SAEs with an embedding-initialized control and a classical one-sparse dictionary-learning baseline.

## 5. Expected contribution

The experiment tests **optimization and dictionary recovery in an unusually clean, identifiable setting**.

A successful result would establish that standard SAE training can recover a known dictionary at realistic LLM vocabulary sizes.

An unsuccessful result would be equally informative: it would demonstrate a gap between the existence of an exact, identifiable solution and the ability of standard training procedures to discover it.

Either outcome would provide a useful baseline for subsequent experiments varying the sparsity parameter \(k\), where the learned dictionary need no longer consist of the original token embeddings.
