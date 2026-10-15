# One-Sparse Autoencoders and Token-Embedding Recovery: A Literature Review

*Research note — 14 October 2026*

## Executive summary

Consider training a **TopK sparse autoencoder (SAE)** directly on the static token embeddings of a pretrained language model. Fix the number of SAE dictionary vectors to the vocabulary size, **`m = V`**, and allow only **one active feature per token**, **`k = 1`**. Will ordinary gradient-based training recover the original token-embedding directions?

The setting has an unusually clear mathematical benchmark: with a **zero decoder bias**, nonnegative one-sparse codes, and **distinct nonzero embedding directions**, exact recovery is both *possible* and *necessary at any zero-reconstruction-error solution*, up to permutation and positive scaling. Nevertheless, the training algorithm need not attain this global optimum. Dead features, unequal token frequencies, initialization, and competition between features could impede recovery.

**Literature finding:** I did not identify a published experiment testing this exact combination of *raw token embeddings*, *TopK `k = 1`*, and *dictionary width `m = V`*. This is a limited search finding, **not proof that no such experiment exists**. The closest work comes from classical one-sparse dictionary learning and spherical clustering, theoretical analyses of dictionary recovery by autoencoders, modern TopK SAE training, and investigations of token-specific SAE features.

The most credible contribution is therefore **an empirical study of optimization and identifiability in a setting with a known global solution**, not a claim that one-sparse dictionary recovery is a new problem.

---

## 1. Background and precise experimental question

A transformer has a vocabulary of size `V` and token embeddings

$$
e_1,\ldots,e_V\in\mathbb R^d.
$$

These are the *static* token embeddings in the embedding matrix, not contextualized residual-stream activations after attention layers.

We train an SAE with `m = V` decoder features (columns of the decoder matrix). A convenient baseline model is

$$
a(x)=W_{\mathrm{enc}}x+b_{\mathrm{enc}},\qquad
z(x)=\operatorname{Top1}(\operatorname{ReLU}(a(x))),
$$

$$
\hat x=W_{\mathrm{dec}}z(x),\qquad
\mathcal L=\sum_{j=1}^{V}p_j\,\|e_j-\hat e_j\|_2^2,
$$

where `p_j > 0` denotes a chosen sampling weight for each token. The decoder bias is **fixed to zero** in this baseline. Top1 retains the largest nonnegative feature activation and sets all other coordinates to zero, so `||z(x)||_0 ≤ 1`.

**Core question.** With random initialization and ordinary gradient-based SAE training, does the learned decoder recover the set of original token-embedding directions, or does optimization systematically fail despite the existence of a perfect solution?

This differs from asking whether an SAE can discover semantically shared features in contextual activations; the raw embedding matrix gives us a directly observable, finite reference dictionary.

### 1.1 The exact-recovery benchmark

Assume that every `e_j` is nonzero and no two token embeddings lie on the same **positive ray**. Choose

$$
v_j=\frac{e_j}{\|e_j\|_2},\quad
W_{\mathrm{dec}}=[v_1\ \cdots\ v_V],\quad
W_{\mathrm{enc}}=W_{\mathrm{dec}}^\top,
$$

with zero encoder and decoder biases. For input `e_j`, the score for feature `j` is `||e_j||_2`. By Cauchy–Schwarz, every other feature scores *strictly less* under the positive-ray assumption. Top1 therefore chooses `j` with activation `||e_j||_2`, and

$$
\hat e_j=\|e_j\|_2v_j=e_j.
$$

Hence **zero reconstruction error is attainable**.

Conversely, suppose each decoder column has unit norm and all examples are reconstructed exactly, still with zero decoder bias and nonnegative one-sparse codes. Every `e_j` must be a positive multiple of one decoder column. Because there are `V` distinct positive rays and only `V` dictionary columns, each ray must receive a distinct column. Thus the dictionary equals the normalized embedding directions **up to permutation**.

This is a simple identifiability observation for this special finite dataset, not a novel general theorem on SAEs. It is conditional: a learned decoder bias changes the target to the directions of `e_j - b_dec`; a signed one-sparse code changes the corresponding equivalence relation; duplicate or collinear embeddings invalidate uniqueness; and approximate low loss does not automatically guarantee good matching.

### 1.2 Why optimization is still nontrivial

The Top1 selection mechanism allocates gradients to only a tiny subset of encoder/decoder features for each example. Training may exhibit:

- **Dead features:** dictionary elements that never win the Top1 competition and receive little or no reconstruction gradient.
- **Feature monopolization:** a few decoder columns initially win for many token embeddings.
- **Frequency effects:** common tokens dominate a frequency-weighted objective while rare-token features remain poorly trained.
- **Initialization dependence:** random starts may give different results even though the zero-error dictionary is identifiable.
- **Encoder–decoder coordination:** selecting the right dictionary column and assigning its optimal coefficient must be learned together.

These optimization phenomena, rather than existence of the optimum, motivate the experiment.

---

## 2. Direct mathematical precedent: one-sparse dictionary learning

### Coates and Ng (2012): *Learning Feature Representations with K-Means*

**Source:** [paper (Stanford PDF)](https://cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf)

Coates and Ng study sparse feature learning through a dictionary `D` and sample-specific sparse codes. One of their objectives is

$$
\min_{D,\{s_j\}}\sum_j\|x_j-Ds_j\|_2^2
\quad\text{subject to }\|s_j\|_0\le 1,\ \|D_i\|_2=1.
$$

They relate this one-sparse formulation to **gain–shape vector quantization** and **spherical k-means**.

**Why it matters:** This is almost the same decoder-side reconstruction objective as the proposed experiment. When the number of dictionary elements equals the number of distinct training directions, assigning one direction per input yields exact reconstruction.

**What it does not answer:** The sparse code in this formulation is optimized directly; it is not necessarily output by a learned affine Top1 encoder trained jointly by backpropagation. Classical clustering algorithms also need not use precisely the nonnegative coefficient convention of a ReLU–TopK SAE. Thus their successes do not show that standard SAE optimization will succeed.

**Takeaway:** The proposed study belongs to the established theory of one-sparse dictionary learning, but examines the behavior of a particular learned encoder and optimization procedure.

**Note:** They only seem to consider the case where the dictionary (number of features) is much smaller than the training dataset (vocabulary size).

---

## 3. Theoretical work on dictionary recovery by autoencoders

### Rangamani et al. (2017): *Sparse Coding and Autoencoders*

**Source:** [arXiv:1708.03735](https://arxiv.org/abs/1708.03735)

This paper asks whether shallow ReLU autoencoders can recover an underlying sparse coding dictionary. It analyses the squared reconstruction-loss gradient near the generating dictionary and proves results on recovering sparse-code support under assumptions on the data-generating process. Synthetic experiments investigate the behavior near the target dictionary.

**Connection:** It gives a principled precedent for studying *when a trainable autoencoder recovers an identifiable reference dictionary*.

**Limit:** It is not a Top1 SAE trained on a finite set of raw token embeddings with one decoder feature per example. Its guarantees and assumptions cannot simply be transferred to the proposed setup.

### Nguyen, Wong and Hegde (2019): *On the Dynamics of Gradient Descent for Autoencoders*

**Source:** [AISTATS / PMLR](https://proceedings.mlr.press/v89/nguyen19a.html)

This paper studies gradient descent in shallow, weight-sharing autoencoders under several generative models, including sparse coding and sparse nonnegative representations. Under appropriate conditions on architecture, hyperparameters, and initialization, it proves recovery of model parameters.

**Connection:** Particularly relevant because it analyzes **optimization dynamics**, not merely existence of a dictionary-recovering solution.

**Limit:** The proofs concern specified probabilistic models and training assumptions, not the particular Top1 architecture or fixed LLM embedding matrices of interest here.

### Nelson, Karaletsos and Locatello (2026): *Toward Identifiable Sparse Autoencoders*

**Source:** [ICML 2026 / PMLR](https://proceedings.mlr.press/v306/nelson26a.html)

This work investigates the instability of SAE dictionaries across training runs, characterizes obstacles to stability, and proposes identifiable SAE variants built from changes to TopK SAE architecture and training. It draws explicit connections to classical dictionary learning and discusses approximate restricted-isometry properties and sparse-code identifiability.

**Connection:** This is the most directly relevant *modern mechanistic-interpretability framing*: reproducibility of learned dictionaries, distinctions between code identifiability and dictionary identifiability, and the impact of training choices.

**Limit:** Its experiments and theoretical claims should not be read as a ready-made convergence theorem for `k = 1`, `m = V` on token embeddings.

---

## 4. TopK autoencoders: architecture and training precedents

### Makhzani and Frey (2013): *k-Sparse Autoencoders*

**Source:** [arXiv:1312.5663](https://arxiv.org/abs/1312.5663)

Introduces an autoencoder that retains only the `k` largest hidden activations. The experiments concern image representation learning, rather than LLM token embeddings.

**Connection:** Establishes the architectural lineage of direct sparsity constraints, including the `k = 1` endpoint.

### Gao et al. (2024; ICLR 2025): *Scaling and Evaluating Sparse Autoencoders*

**Source:** [arXiv:2406.04093](https://arxiv.org/abs/2406.04093) · [ICLR publication](https://proceedings.iclr.cc/paper_files/paper/2025/hash/42ef3308c230942d223c411adf182c88-Abstract-Conference.html)

Develops a modern TopK SAE training recipe, evaluates reconstruction–sparsity trade-offs and scaling, and addresses **dead latents**, including the use of auxiliary losses and improved initialization techniques. The principal experiments train on contextual LLM residual-stream activations.

**Connection:** A practical blueprint for the neural architecture, training diagnostics, and dead-feature mitigations required in the proposed experiment.

**Limit:** This is not the `k = 1`, `m = V`, *raw embedding matrix* experiment. It is also important not to import dead-latent recovery auxiliaries into the first baseline without documenting how they modify the objective.

---

## 5. Token-specific SAE features and the embedding connection

### Cho et al. (2026): *Are Single-Token Sparse Autoencoder Features Causally Necessary? Layer-Depth and SAE-Family Effects*

**Source:** [arXiv:2607.20596](https://arxiv.org/abs/2607.20596)

Cho and colleagues examine features that preferentially activate for individual vocabulary tokens across multiple models and SAE families. Their analysis studies where such features occur and tests their causal consequences through ablation. They report that single-token features are especially concentrated in early layers and that causal results depend strongly on SAE training family.

**Connection:** Empirical evidence that token-specific SAE features arise in contemporary interpretability pipelines, and a warning that training recipes influence apparently intrinsic learned features.

**Limit:** The features are analysed in existing SAEs trained on model activations; this does **not** test recovery of all static token embeddings by a purpose-built `k = 1`, `m = V` SAE.

### Subramanian et al. (2018): *SPINE: SParse Interpretable Neural Embeddings*

**Source:** [AAAI 2018](https://mlanthology.org/aaai/2018/subramanian2018aaai-spine/)

SPINE trains sparse autoencoding representations of pre-existing word embeddings such as GloVe and word2vec, emphasizing interpretable dimensions and downstream performance.

**Connection:** Establishes that applying sparse autoencoders **directly to static lexical embeddings** is not new.

**Limit:** The architecture and sparsity penalties are not the `k = 1`, `m = V` token-by-token dictionary-recovery experiment. Its stated goal is interpretable distributed representations rather than reproduction of the original input directions.

---

## 6. An informal but unusually close discussion

### Heimersheim and Engels (LessWrong shortform discussion, 2024–2025)

**Source:** [discussion thread](https://www.lesswrong.com/posts/ZBjhp6zwfE8o8yfni/stefanhex-s-shortform)

Josh Engels asks whether an optimally trained `k = 1` TopK SAE should recover a clustering-based dictionary. Stefan Heimersheim agrees with that expectation and describes using learned centroids as encoder/decoder directions. The ensuing informal experiments compare clustering and SAE reconstructions on contextual GPT-2/Pythia activations.

**Connection:** This discussion directly anticipates the relationship between `k = 1` SAEs and clustering. It is valuable evidence that the *conceptual comparison* is already familiar within mechanistic interpretability.

**Limits:** Informal posts are not peer-reviewed experimental studies, and the reported data are **contextual activations**, not the static embedding table with one feature per token. The reported runs therefore do not resolve the proposed experiment.

---

## 7. How close are these works to the proposed experiment?

| Work | One-sparse (`k = 1`) focus? | Raw token/word embeddings? | One feature per vocabulary entry (`m = V`)? | Main relevance |
|---|---|---|---|---|
| Coates & Ng (2012) | **Yes**, in a dictionary-learning formulation | No | Not the specific design | Mathematical optimization analogue |
| Rangamani et al. (2017) | Not specifically | No | No | Autoencoder dictionary-recovery theory |
| Nguyen et al. (2019) | Not specifically | No | No | Gradient-descent recovery guarantees |
| Makhzani & Frey (2013) | Supports general `k`, including 1 | No | No | TopK architectural origin |
| SPINE (2018) | No | **Yes**, static word embeddings | No | Closest precedent in input type |
| Gao et al. (2024/25) | General TopK, not this exact endpoint | No | No | Modern training and dead-latent methods |
| Nelson et al. (2026) | General TopK | No | No | SAE stability and identifiability |
| Cho et al. (2026) | Studies *single-token features*, not `k = 1` codes | No, contextual activations | No | Token-specific feature phenomenon |
| Heimersheim/Engels discussion | **Yes**, explicitly discusses `k = 1` | No, contextual activations | No | Clustering versus neural SAE |

**Interpretation:** No entry in this set demonstrates the exact experiment. The design occupies the intersection of established one-sparse dictionary learning, static lexical embedding analysis, and modern TopK SAE optimization.

---

## 8. A focused experimental programme

### Primary hypothesis

With `m = V`, `k = 1`, zero decoder bias, and unit-normalized decoder columns, a standard Top1 SAE trained on all static token embeddings should approach the recoverable one-to-one dictionary, *provided training overcomes feature-allocation and optimization problems*.

The existence and uniqueness of the zero-error dictionary are analytical benchmarks, **not experimental claims**.

### Baseline comparisons

1. **Constructed exact solution.** Set decoder directions equal to normalized embeddings and the encoder to their transpose. Verify zero loss numerically. This is a correctness test, not evidence that optimization succeeded.
2. **One-sparse dictionary-learning baseline.** Optimize codes directly or use a spherical-clustering/gain–shape method, recording its coefficient and sign conventions. Compare optimization quality with the neural encoder.
3. **Randomly initialized Top1 SAE.** Train independently across random seeds with frozen zero decoder bias and normalized decoder vectors.
4. **Controlled optimization variants.** Evaluate encoder–decoder initialization, dead-feature interventions, and balanced versus frequency-weighted sampling as separate ablations.

### Core measurements

- **Reconstruction:** mean squared error across all `V` embeddings, not only sampled training batches; also report the worst-case and percentile errors.
- **One-to-one dictionary recovery:** after normalizing, solve a **global bipartite assignment** between decoder directions and original embedding directions using cosine similarity, then report distribution of matched cosines and percentages above predeclared thresholds. A nearest-neighbour maximum alone can count the same decoder repeatedly and overstate recovery.
- **Feature usage:** number of decoder features winning at least once; count of unused/dead features; tokens per winning feature; training evolution of these statistics.
- **Selection accuracy:** compare the Top1 feature chosen by the trained encoder against the best one-to-one matched feature for every token.
- **Robustness:** variation across seeds, optimizer settings, initialization, and token-sampling distributions.

**Important implementation caveat:** Finding the globally best one-to-one matching at vocabulary scale (`V ≈ 50,000`) is computationally expensive if done naively. Report computational approximations transparently (e.g., sparse candidate matching) and validate them on smaller controlled cases.

### Essential controls

- Fix `b_dec = 0` in the primary experiment; otherwise exact reconstruction could recover directions measured from an arbitrary learned origin.
- Specify whether Top1 uses `ReLU` **before** the top-one selection; enforce nonnegative codes consistently in both theory and implementation.
- Verify that embedding directions are distinct and nonzero, at least numerically to tolerance; remove duplicates or discuss them separately.
- Decide whether `p_j` is **uniform over vocabulary entries** or estimated from a text corpus. A uniform objective is cleaner for recovery; frequency weighting tests realistic training imbalance.
- Avoid conflating a zero reconstruction loss obtained by **constructed initialization** with convergence from random initialization.
- Observe memory and compute costs: a width-`V` encoder and decoder each require order `Vd` parameters; evaluating all-token Top1 scores is expensive.

### What would count as a useful outcome?

- **Reliable recovery:** Establishes a strong practical baseline showing that current SAE optimizers can find a known dictionary at realistic vocabulary scale.
- **Frequent partial recovery:** Quantifies how dead features, sampling imbalance, and initialization obstruct an identifiable optimum; tests mitigation strategies.
- **Very low loss but imperfect matching:** Motivates an analysis of near-optimal solutions and how reconstruction error relates quantitatively to dictionary alignment.

Even if recovery succeeds, the study remains valuable as a **controlled benchmark** for later experiments with `k > 1`, where multiple features can contribute to the same token and uniqueness is no longer automatic.

---

## 9. Conclusions and research positioning

The proposed experiment should be pitched modestly but precisely:

> We investigate whether standard TopK sparse autoencoder training recovers the token-embedding dictionary in the one-sparse, vocabulary-width regime. Unlike ordinary SAE interpretability experiments, this setting has a known exact-reconstruction solution and (under transparent assumptions) an identifiable global dictionary. It therefore isolates the optimization and feature-allocation aspects of SAE training from the ambiguity of evaluating features on contextual activations.

This is **not** a new theory of dictionary recovery and is **not** a demonstration that ordinary SAEs automatically discover unique semantic concepts. It is a small, clean empirical test linking classical one-sparse dictionary learning to present-day SAE training practice.

The open literature search reported here did **not** locate the exact raw-embedding/Top1/`m = V` experiment; this is a statement about the search, not proof of nonexistence or a priority claim.

---

## References and reading order

**Start here (closest to the core experiment):**

1. **Coates, A., & Ng, A. Y. (2012).** *Learning Feature Representations with K-Means.* [PDF](https://cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf). One-sparse dictionary-learning formulation and clustering connection.
2. **Nguyen, T. V., Wong, R. K. W., & Hegde, C. (2019).** *On the Dynamics of Gradient Descent for Autoencoders.* AISTATS, PMLR 89. [Paper](https://proceedings.mlr.press/v89/nguyen19a.html). Gradient-descent recovery theory.
3. **Nelson, W., Karaletsos, T., & Locatello, F. (2026).** *Toward Identifiable Sparse Autoencoders.* ICML, PMLR 306. [Paper](https://proceedings.mlr.press/v306/nelson26a.html). Modern SAE dictionary identifiability and stability.
4. **Gao, L., et al. (2024; published 2025).** *Scaling and Evaluating Sparse Autoencoders.* ICLR 2025. [Paper](https://arxiv.org/abs/2406.04093). TopK training and dead-latent practicalities.

**Further background and context:**

5. **Rangamani, A., et al. (2017).** *Sparse Coding and Autoencoders.* [arXiv](https://arxiv.org/abs/1708.03735). Reconstruction-loss landscape near a known dictionary.
6. **Makhzani, A., & Frey, B. (2013).** *k-Sparse Autoencoders.* [arXiv](https://arxiv.org/abs/1312.5663). Original k-sparse autoencoder architecture.
7. **Subramanian, A., et al. (2018).** *SPINE: SParse Interpretable Neural Embeddings.* AAAI. [Paper](https://mlanthology.org/aaai/2018/subramanian2018aaai-spine/). SAE-like sparsification of existing word embeddings.
8. **Cho, S., et al. (2026).** *Are Single-Token Sparse Autoencoder Features Causally Necessary? Layer-Depth and SAE-Family Effects.* [arXiv](https://arxiv.org/abs/2607.20596). Single-token features in contextual SAEs.
9. **Heimersheim, S., & Engels, J. (2024–2025 discussion).** *StefanHex's Shortform*, LessWrong. [Thread](https://www.lesswrong.com/posts/ZBjhp6zwfE8o8yfni/stefanhex-s-shortform). Informal discussion of Top1 SAE and clustering correspondence.

*Scope note: This review is standalone and focuses on the `k = 1`, `m = V` investigation. It is not an exhaustive survey of sparse autoencoders, word embeddings, or clustering algorithms.*
