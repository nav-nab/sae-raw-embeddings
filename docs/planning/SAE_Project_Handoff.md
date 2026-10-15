# Project handoff: Can a one-sparse SAE recover a language model's token embeddings?

**Prepared:** 15 October 2026  
**Status:** Research plan and literature reconnaissance; experiment not yet implemented or run in this conversation.  
**Audience:** An ML research/coding agent joining the project without access to the original discussion.  
**Scope:** The substantive discussion about sparse autoencoders (SAEs), one-sparse dictionary learning, token-embedding geometry, prior work, and the planned experiment. This is a synthesis of the discussion, not a verbatim transcript or a claim of exhaustive external literature verification.

## 0. Executive summary

**Central question:** Train a standard TopK sparse autoencoder directly on the *static input-token embedding matrix* of a pretrained language model, with **one active feature per token (`k = 1`)** and **exactly as many dictionary features as vocabulary tokens (`m = V`)**. Starting from random initialization, does gradient-based optimization recover the original token-embedding directions?

Initial target: **GPT-2 Small**, vocabulary size `V = 50,257`, embedding dimension `d = 768`. Treat its `V` rows as `V` training vectors, not as contextual activations sampled from passages of text.

The test is appealing because the desired solution is known in advance. With a fixed **zero decoder bias**, **unit-norm decoder columns**, **nonnegative one-sparse codes**, and embeddings lying on **pairwise distinct positive rays**, the normalized embedding directions form a *zero-loss dictionary*, and every zero-loss dictionary with `m = V` must be that dictionary up to permutation. What is unknown is whether a realistically trained Top1 SAE reliably reaches it. Possible barriers: dead features, competition for tokens, initialization, uneven token frequencies, and encoder/decoder coordination.

**Literature-search outcome so far:** We did **not** locate a documented implementation satisfying all three defining conditions simultaneously: **(i) raw/static LLM token embeddings as the training data, (ii) a learned TopK SAE with `k = 1`, and (iii) dictionary width `m = V`**. This is **not proof of novelty**. Classical one-sparse dictionary learning already studies a closely related reconstruction objective; several SAE papers study token embeddings, vocabulary-aligned features, or TopK models, but not this precise combination.

**Recommended immediate task:** Build a reproducible PyTorch baseline, validate the exact constructed solution and matching metrics on a small sub-vocabulary, then scale to the full GPT-2 vocabulary across multiple random seeds. Treat computational matching costs and feature starvation as first-class concerns.

---

## 1. Research motivation and terminology

The larger intellectual context is **mechanistic interpretability** and the **superposition hypothesis**: models may encode more useful features than the ambient representation dimension by exploiting sparsity and nearly orthogonal feature directions. Sparse autoencoders are used to seek an interpretable overcomplete dictionary for such representations.

The conversation distinguished several notions that should **not** be conflated:

- **Raw or static token embedding:** a fixed row `e_i` of the model's token embedding table, prior to contextual transformer computation. The dataset here is the finite set of embedding-table rows; one row corresponds to one vocabulary *token ID*, not necessarily one linguistic word.
- **Contextual representation / residual-stream activation:** an activation computed at a particular sequence position after some or all transformer computations. Its value depends on context. Most mechanistic-interpretability SAEs are trained on these activations; **that is not the proposed dataset**.
- **SAE feature:** typically one decoder column together with an associated encoder coordinate; it is not automatically a vocabulary token or a linguistic concept.
- **Single-token feature** in the literature: an SAE feature that preferentially activates for a particular vocabulary token. This observational property is **not equivalent** to setting the SAE sparsity hyperparameter to `k = 1`.
- **`m = V`:** the number of decoder columns equals the number of vocabulary entries. This does *not* mean the embedding dimensionality is `V`: the embedding vectors still lie in `R^d` with `d << V`.

The user wanted a foundational *optimization/recovery* benchmark, **not** another loosely related demonstration of semantic feature discovery. In literature searches, an exact match must satisfy all three defining conditions, rather than only one or two.

## 2. Mathematical formulation

### 2.1 Dataset

Let `E` be the input-token embedding matrix of the frozen pretrained model, with rows

$$
e_1,\ldots,e_V \in \mathbb R^d, \qquad E \in \mathbb R^{V\times d}.
$$

For GPT-2 Small, `V = 50,257` and `d = 768`. Extract the **input embedding table** (often named `wte`, or `transformer.wte.weight` in standard GPT-2 implementations); do not accidentally use positional embeddings, a layer activation, or tokenized text samples.

Each training example is a distinct `e_i`. **Uniform sampling over vocabulary IDs** is the preferred primary experiment: the goal is recovery of *all* directions. Frequency-weighted sampling can be introduced later as a controlled stress test.

### 2.2 SAE architecture

One explicit baseline definition is

$$
a(x) = W_{\mathrm{enc}}x+b_{\mathrm{enc}},
$$

$$
z(x) = \operatorname{Top1}(\operatorname{ReLU}(a(x))),
$$

$$
\hat x=W_{\mathrm{dec}}z(x), \qquad
W_{\mathrm{dec}}=[v_1\ \cdots\ v_m]\in\mathbb R^{d\times m}.
$$

Here `Top1` keeps the largest **positive** activation and sets all other coordinates to zero. If none is positive, `z = 0`; thus `||z||_0 <= 1` rather than necessarily exactly one. State and test the implementation's tie-handling and behavior with nonpositive inputs.

Set

$$
m=V, \qquad k=1, \qquad \|v_j\|_2=1\ \text{for all }j, \qquad b_{\mathrm{dec}}=0.
$$

The primary loss is mean squared reconstruction error,

$$
\mathcal L = \frac{1}{V}\sum_{i=1}^{V}\|e_i-\hat e_i\|_2^2.
$$

No separate `L^1` sparsity penalty is needed: Top1 enforces sparsity directly. Train both encoder and decoder unless explicitly running a control. Enforce unit decoder norms throughout training, preferably by a clearly documented projected/normalized optimization rule; do not normalize *only* for final evaluation. Keep decoder bias *fixed*, not trainable, in the primary experiment.

A minor architectural choice requiring an explicit decision is whether to use the **independent affine encoder** written above or a **tied encoder** `W_enc = W_dec^T`. The main proposed experiment concerns a standard trainable SAE with an independent encoder; a tied variant is a useful ablation.

### 2.3 Constructed exact optimum

Assume each `e_i` is nonzero and no two embeddings lie on the same **positive ray** (no `e_i = c e_j` with `c > 0` for `i != j`). Set

$$
v_i=\frac{e_i}{\|e_i\|_2}, \qquad W_{\mathrm{enc}}=W_{\mathrm{dec}}^\top, \qquad b_{\mathrm{enc}}=0.
$$

Then, on input `e_i`, feature `i` receives score `||e_i||_2`. Every other feature has score strictly less, by Cauchy–Schwarz and the distinct-positive-ray assumption. Therefore Top1 picks `i` with coefficient `||e_i||_2`, giving

$$
\hat e_i = \|e_i\|_2 v_i = e_i.
$$

This establishes **existence of a zero-error SAE solution** with the chosen architecture. The solution is not a discovery by optimization: using this construction to initialize the network is only a **sanity check / positive control**.

### 2.4 Identifiability of the zero-loss dictionary

Conversely, assume **nonnegative** one-sparse codes, zero decoder bias, unit-norm decoder columns, and exact reconstruction of all `V` embeddings. Every `e_i` must be a positive scalar multiple of some decoder column `v_j`. The `V` inputs occupy `V` distinct positive rays and only `V` columns are available. Consequently each ray requires its own column, and

$$
\{v_1,\ldots,v_V\} =
\left\{\frac{e_1}{\|e_1\|_2},\ldots,\frac{e_V}{\|e_V\|_2}\right\}
$$

**as sets, up to a permutation**. The weights of the encoder need not be unique. This is a simple geometric observation for a finite dataset, **not** a general SAE-identifiability theorem.

Critical caveats:

- If the decoder bias is learnable, perfect reconstruction may instead be based on directions `e_i - b_dec`; the original directions are not identified by the same argument.
- If one-sparse coefficients may be **negative**, recovery is to *unoriented lines*, not positive rays, and possible sign flips must be accounted for.
- Exact duplicate/positive-collinear embeddings break the one-to-one uniqueness argument. Check numerically and record any degeneracies.
- **Near-zero loss does not by itself prove near-exact matching.** Quantitative stability depends on angular separation, vector norms, and the achieved errors. Measure recovery explicitly.
- If optimization permits zero active features, that output cannot reconstruct any nonzero token embedding exactly; this does not affect the zero-loss argument.

### 2.5 Why recovery might fail under training

The global solution is trivial to *construct* but may be difficult to *learn*. Top1 sends reconstruction gradients primarily through winners, so columns that rarely win can remain undertrained or completely dead. A few columns may monopolize many tokens. Independently learned encoder and decoder parameters must coordinate both assignments and amplitudes. Random initialization and learning-rate choices may matter greatly. Sampling by corpus token frequency could starve rare vocabulary entries, even though every entry is required by the theorem.

The true experimental question is therefore **the gap between an identifiable global solution and the solution found by ordinary neural optimization**.

---

## 3. SAE and dictionary-learning concepts discussed

### 3.1 TopK versus `L^1` sparsity

An `L^1`-regularized SAE minimizes reconstruction error plus a term such as `lambda * ||z||_1`, encouraging sparse activations (soft-thresholding appears in related proximal formulations). A **TopK SAE** enforces a hard budget of at most `k` nonzero features by selecting the largest activations. At `k = 1`, every reconstructed example lies on at most one decoder ray if activations are nonnegative. These are different objectives/optimization dynamics and must not be treated as interchangeable baselines without stating the difference.

### 3.2 One-sparse dictionary learning and clustering

The classical reconstruction problem

$$
\min_{D,\{s^{(i)}\}}\sum_i\|x^{(i)}-Ds^{(i)}\|_2^2
\quad\text{subject to}\quad
\|s^{(i)}\|_0\le1,\quad \|D_j\|_2=1
$$

is a close mathematical analogue. Related algorithms optimize a dictionary and **separate per-example codes**; a standard SAE instead learns an **amortized neural encoder** that produces codes from `x`. Classical formulations may allow signed coefficients; the ReLU–Top1 SAE does not. Spherical k-means and gain–shape vector quantization are pertinent but should be compared at the level of the actual objective and coefficient convention.

A notation trap when reading **Coates & Ng (2012)**: their `n` denotes input dimension (`d` here), their `m` denotes number of examples (`V` here), and their `k` denotes number of dictionary elements (`m` here). Their `k` is **not** the TopK activation count. In their notation the corresponding equal-size case is `k = m`. Their paper does not appear to test the exact one-dictionary-atom-per-example vocabulary setting.

### 3.3 Nonorthogonality, superposition, and spherical codes

Decoder features **are not generally constrained to be orthogonal**, and at `m = V > d` they cannot all be orthogonal. In the exact `k = 1` solution their pairwise cosine similarities are simply those of the original embedding directions.

For any `m > d` unit vectors, the **Welch bound** gives

$$
\max_{i\ne j}|\langle v_i,v_j\rangle|^2\ge
\frac{m-d}{d(m-1)}.
$$

For `d = 768`, `m = 50,257`, this implies maximum absolute off-diagonal cosine at least approximately `0.0358`. This is only a lower bound, not a prediction of actual GPT-2 embedding geometry.

More generally, spherical/projective coding quantifies how many vectors can have small pairwise (absolute) correlations. The conversation noted that the Welch bound becomes weak for large dimensions relative to `1/epsilon^2`, while probabilistic constructions allow exponentially many approximately orthogonal vectors at fixed tolerance. A schematic superposition interference formula is

$$
\langle x,v_j\rangle = z_j + \sum_{i\ne j}z_i\langle v_i,v_j\rangle,
\qquad x=\sum_i z_i v_i.
$$

For the present **Top1** setup the sparse-code recovery problem is much simpler than for `k > 1`; broader superposition questions motivate later extensions but are not themselves tested by the initial experiment.

---

## 4. Planned experimental protocol

### 4.1 Minimum viable study

1. **Extract embeddings.** Load frozen GPT-2 Small input embedding matrix; save model revision, library versions, shape, dtype, embedding norms, any near-duplicate direction diagnostics, and random seeds.
2. **Write a mathematically faithful SAE.** Width `m = V`, ReLU–Top1 encoder, zero decoder bias, decoder columns normalized every training step, MSE loss. Implement consistent handling of zero/negative preactivations and ties.
3. **Positive control.** Set `W_dec` to normalized embedding rows and `W_enc = W_dec^T` with zero encoder bias. Verify nearly zero reconstruction error for the *entire* vocabulary. This tests data extraction and implementation, not optimization.
4. **Random-start runs.** Train from several independent initializations, saving optimization curves and feature-usage statistics. Specify optimizer, schedule, batch size, normalization, initialization scales, and total number of passes through the vocabulary.
5. **Measure dictionary recovery.** Compare learned unit decoder columns with unit embedding vectors under an approximate or exact **one-to-one assignment**. Report the distribution of matched cosines, not just reconstruction error.
6. **Compare baselines.** At least a classical one-sparse dictionary-learning / spherical-clustering optimizer, with sign and coefficient conventions aligned where possible. Run controlled ablations only after the unmodified baseline is recorded.

**Recommended staging:** Debug on a small synthetic dictionary where exact recovery is known, then a small real-vocabulary subset, then full GPT-2 Small. With a subset of `n` tokens, use `m = n`, not the full width, when testing the equal-size claim.

### 4.2 Measurements and diagnostics

| Quantity | How to measure / why it matters |
|---|---|
| Mean reconstruction loss | Compute across **all** tokens at evaluation time; do not infer from sampled minibatches alone. |
| Error distribution | Mean, median, percentiles, worst-case per-token error, ideally normalized errors as well. |
| Decoder recovery | Globally matched cosine similarities of learned unit decoder columns to unit token embeddings; threshold fractions. |
| Feature utilization | Number of distinct winning features; unused features; number of tokens assigned to each feature; trajectories over training. |
| Encoder assignment accuracy | After matching decoder columns to tokens, fraction of tokens whose encoder picks its matched feature. |
| Stability | Variation across random seeds, optimizer hyperparameters, and initialization schemes. |
| Token frequency sensitivity | Compare **uniform token-ID sampling** with corpus-frequency sampling in a clearly separate experiment. |
| Computational cost | Parameter/memory footprint, training time, and matching time; include practicality at `V ~ 50,000`. |

**Matching pitfall:** Nearest-neighbour cosine matching alone can pair *many* embeddings to the same decoder column, substantially overestimating recovery. Prefer a global permutation/bipartite assignment. A dense `V × V` cosine matrix has approximately **2.53 billion entries**, about **10.1 GB in float32** before matching overhead; general dense assignment is also computationally expensive. Use blockwise similarity search with sparse candidate matches or validated approximations at full scale, and compare against exact matching on small subproblems. Distinguish *nearest-neighbour alignment* from *verified one-to-one recovery* in all reports.

**Compute footprint:** At `m = V`, an untied affine encoder and decoder have roughly `2dV ≈ 77.2 million` weight parameters (excluding biases). FP32 weights alone take approximately 309 MB; gradient buffers, Adam optimizer states, Top1 scores (`batch × V`), and evaluation/matching significantly increase requirements. A consumer GPU or optimized batched CPU training might be feasible but needs profiling; this is not a promise of a particular runtime.

### 4.3 Experimental arms / ablations

- **Constructed exact solution:** sanity check, not learned recovery.
- **Default SAE from random initialization:** primary result.
- **Embedding-seeded initialization:** how much does having a good initial dictionary help? Not a fair stand-in for random-start recovery.
- **Tied versus untied encoder:** does amortization/coordination cause avoidable failures?
- **Explicit-code / one-sparse clustering baseline:** isolate dictionary optimization from neural encoder training.
- **Dead-latent interventions:** e.g. resampling/reinitialization or auxiliary loss, clearly indicated as changes to the training recipe; avoid silently altering the baseline objective.
- **Uniform versus frequency-weighted tokens:** test whether skewed data frequency harms recovery.
- **Later:** vary `k` (`k > 1`), widths `m`, token subsets, embedding dimensionality, and model families. Uniqueness generally changes when `k > 1`.

### 4.4 Success, partial success, and negative results

- **Reliable near-perfect recovery:** demonstrates that a chosen practical optimization recipe can find a known dictionary at realistic vocabulary width.
- **Loss decreases but many dictionary features stay dead:** demonstrates allocation/optimization failure; the global optimum still exists.
- **Very low reconstruction loss but poor global matching:** investigate rare failures, minimum angular margins, scale-sensitive errors, and whether assignment approximations are trustworthy; don't equate approximate reconstruction with exact dictionary identification.
- **Seed-dependent results:** potentially a publishable empirical characterization of stability/failure modes if carefully benchmarked.

Do **not** claim that the mathematics predicts SGD success, that learned features must be semantically interpretable, or that the experiment establishes novelty of one-sparse dictionary learning itself.

---

## 5. Literature reconnaissance: relevant findings, precisely bounded

The discussion involved several rounds of searching, including an explicit deeper search for the **exact** experiment in papers, code, notebooks, and research forums. Its result was a **negative search finding**, not proof of nonexistence. The following entries explain why the closest candidates did not qualify. Links are supplied as starting points for the next agent's independent verification.

### 5.1 Highest-priority intellectual antecedents

**Coates & Ng (2012), _Learning Feature Representations with K-Means_.**  
[Stanford PDF](https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf)

Studies one-sparse dictionary learning, gain–shape vector quantization, and related clustering. The decoder-side objective is very close; however, codes are optimized rather than generated by a learned Top1 encoder, and the paper is not an experiment on LLM raw embeddings with one atom per token. **This is the first paper to understand mathematically before implementing baselines.** Check its local notation carefully (see §3.2).

**Rangamani et al. (2017), _Sparse Coding and Autoencoders_.**  
[arXiv](https://arxiv.org/abs/1708.03735)

Studies autoencoder recovery of a ground-truth sparse-coding dictionary under assumptions; relevant for support recovery and the optimization landscape. Not the finite embedding-table Top1 experiment.

**Nguyen, Wong & Hegde (2019), _On the Dynamics of Gradient Descent for Autoencoders_.**  
[PMLR](https://proceedings.mlr.press/v89/nguyen19a.html)

Analyzes gradient-descent recovery under generative assumptions and particular architectures. Directly relevant to the gap between existence of a solution and optimization; not a plug-in guarantee for this setting.

**Nelson, Karaletsos & Locatello (2026), _Toward Identifiable Sparse Autoencoders_.**  
[PMLR](https://proceedings.mlr.press/v306/nelson26a.html)

Modern work on SAE identifiability/stability and architecture/training choices. Related motivation but not known to run the exact configuration.

**Gao et al. (2024/2025), _Scaling and Evaluating Sparse Autoencoders_.**  
[arXiv](https://arxiv.org/abs/2406.04093)

A key practical source for modern TopK training, dead latents, sparsity budgets, and evaluation. Uses contextual activations rather than the target raw-embedding dataset.

**Makhzani & Frey (2013), _k-Sparse Autoencoders_.**  
[arXiv](https://arxiv.org/abs/1312.5663)

Early architectural precedent for retaining only `k` active units.

### 5.2 Closest work on token embeddings / token-specific features

**SPINE (2018), _SParse Interpretable Neural Embeddings_.**  
[AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/11935/11794)

Trains sparse autoencoding representations from static word embeddings (such as word2vec/GloVe), but uses a different architecture/sparsity objective and does not set `k = 1, m = V` for direction recovery.

**_Word Equations_ (2020).**  
[arXiv](https://arxiv.org/abs/2004.13847)

Sparse-code analysis of word embeddings with a different fixed/structured dictionary approach; not the exact trained Top1 SAE.

**Cho et al. (2026), _Are Single-Token Sparse Autoencoder Features Causally Necessary?_**  
[arXiv](https://arxiv.org/abs/2607.20596)

Analyzes *existing* SAEs trained on contextual residual-stream activations, identifying features that fire strongly for particular tokens; does not train a vocabulary-width Top1 SAE on raw embedding-table rows. The discussion cited a reported **mean cosine ~0.674** between single-token SAE decoder features and associated token embeddings versus **~0.392** for a comparison group; these are *feature–embedding alignments in their contextual SAE setup*, **not** recovery scores in our proposed setup. Their study of a token feature's top activating tokens must not be mistaken for SAE sparsity `k = 1`.

**VASAE (2026), vocabulary-aligned SAEs.**  
[arXiv](https://arxiv.org/abs/2606.27941)

Uses vocabulary-sized dictionary `m = V` and an alignment concept, but TopK budget is reported as `k = 32` and inputs are contextual activations (with alignment regularization), not a raw-token-embedding Top1 reconstruction test. Particularly tempting false match because it satisfies `m = V`.

**_Beyond a Bag of Features_ (2026).**  
[arXiv](https://arxiv.org/abs/2608.11197)

Reportedly applies already-trained SAEs to raw embedding inputs rather than freshly training a Top1 SAE on those vectors. Distinguish **probing an embedding table with an SAE** from **training the SAE on the embedding table**; verify the precise section if citing.

**Hebbian Sparse Autoencoder (2025, ICLR workshop).**  
[OpenReview](https://openreview.net/forum?id=2ZoozRf41d) · [author follow-up](https://www.lesswrong.com/posts/7qjGWJcRSfMga46uR/hebbian-learning-through-the-lens-of-sae-traning)

The paper's abstract mentions token embeddings, making it unusually promising. A subsequent account discussed 64-dimensional TinyStories residual activations, a 3,200-feature dictionary, and many active features per input, **not** the target `k = 1, m = V` experiment. Because the abstract and follow-up may describe differing experiments, **the paper itself warrants a careful full-text check before making categorical claims**. Regardless, no verified exact match was found in the previous search.

**Heimersheim / Engels informal LessWrong discussion.**  
[Thread](https://www.lesswrong.com/posts/ZBjhp6zwfE8o8yfni/stefanhex-s-shortform)

Explicitly discusses Top1 SAE versus clustering and informal tests with contextual GPT-2/Pythia activations. Confirms that the idea of Top1 as clustering is not new; does not resolve the raw embedding `m = V` benchmark.

### 5.3 Other papers discussed as context, not exact precedents

- **Conjecture (2023), _Taking Features Out of Superposition with Sparse Autoencoders_** — synthetic feature-recovery experiments, generally `L^1`-sparse SAE; [research note](https://www.conjecture.dev/research/interim-research-report-taking-features-out-of-superposition-with-sparse-autoencoders).
- **Anthropic (2022), _Toy Models of Superposition_** — framing of sparse features and superposition; [article](https://transformer-circuits.pub/2022/toy_model/index.html).
- **Paulo & Belrose (2025/2026), _Sparse Autoencoders Trained on the Same Data Learn Different Features_** — cross-seed feature stability; [arXiv](https://arxiv.org/abs/2501.16615).
- **Chanin et al. (2024), _A is for Absorption_** — feature absorption and imperfect SAE decomposition; [arXiv](https://arxiv.org/abs/2409.14507).
- **Balagansky et al. (2025), hierarchical sparsity-budget SAE** — variants supporting multiple `k`; [ACL](https://aclanthology.org/2025.emnlp-main.515/).
- **Matryoshka visual SAEs (2025)** — experiments varying `k`, but on visual activations; [arXiv](https://arxiv.org/abs/2504.02821).
- **Jacquier et al. (2026), sparse-budget robustness** — [arXiv](https://arxiv.org/abs/2606.27321).
- **Huang et al. (2026), rare feature sensitivity** — [arXiv](https://arxiv.org/abs/2609.37857).
- **_Beyond Label Attention_ (2024)** — sparse medical text embedding analysis; [arXiv](https://arxiv.org/abs/2411.00173).

**Literature-search mandate if continued:** Search specifically for *training* a learned SAE on **embedding matrix rows**, **Top1 / one positive activation**, and **dictionary width equal to vocabulary size**. Check methods/code, not merely abstracts. Do not return a catalogue of broadly related papers as if it answered the novelty question.

---

## 6. Decisions already made versus questions still open

### 6.1 Core specification agreed in discussion

- Data: GPT-2 Small **static input token embeddings**, each row as an example.
- Baseline architecture: trainable TopK SAE with **`k = 1`**, **`m = V`**, **nonnegative codes**.
- Decoder: columns constrained to unit norm **during** training; **zero fixed decoder bias**.
- Baseline loss: MSE, no extra `L^1` penalty.
- Initialization for the principal test: **random**, with multiple seeds; constructed exact dictionary is a separate control.
- Main questions: reconstruction error, recovery of embedding directions under permutation, dead/unused features, and stability.
- Classical one-sparse dictionary learning is the primary comparative baseline.

### 6.2 Open methodological decisions the next agent should resolve/document

1. **Top1 activation convention:** ReLU then select largest positive activation; define no-positive behavior and ties. Verify gradients precisely (including normalization treatment).
2. **Encoder design:** independent vs tied; bias initialization; scaling. Record all choices and avoid accidental data-dependent initialization in random-start tests.
3. **Feature-death mitigation:** primary run *without* corrective extras, then clearly labeled ablations with resets/resampling or auxiliary losses.
4. **Optimizer and schedule:** SGD vs Adam/AdamW; learning rates, weight decay, batch size, training duration, initialization and normalization scheme. Monitor interaction with the hard winner-take-all operation.
5. **Matching at scale:** exact vs candidate-restricted bipartite assignment; correctness checks for candidate coverage.
6. **Embedding norms:** raw versus norm-standardized inputs. **Primary theory applies to raw vectors with nonnegative scalar coefficients**; normalizing inputs alters optimization weighting and should be a separate comparison.
7. **Sampling:** uniform token IDs for recovery; corpus-frequency weighting only as a controlled variant.
8. **Check separability:** document duplicate/near-identical token embedding directions and the numerical angular margin.
9. **Evaluation protocol:** no train/test split is necessary for the mathematical recovery benchmark (the finite dictionary itself is the object of recovery), but report all-token evaluation and any supplementary generalization experiments separately.
10. **Novelty claims:** do not claim nobody has done it; the prior search was negative, not exhaustive.

## 7. Suggested deliverables for the next agent

**A. Reproducible codebase**

- Data-loading module for GPT-2 Small embeddings, revision pinning, checksum/metadata.
- Correct and well-tested Top1 SAE with normalized decoder columns and fixed zero decoder bias.
- Training script configurable by seeds, width, vocabulary subset, initialization, optimizer, and schedule.
- Evaluation and diagnostic module with blockwise cosine search and validated one-to-one matching.
- Baseline classical dictionary-learning/clustering routine with documented coefficient rules.
- Unit tests covering exact construction, Top1 behavior, all-feature usage, normalization, and matching correctness.

**B. Experimental report**

- Baseline training curves (loss, dead features, unique assignments).
- Reconstruction error and cosine-recovery distributions, including worst cases and seed variability.
- Comparison to constructed exact solution and classical dictionary-learning baseline.
- Ablations isolating initialization, tied/untied encoder, feature resampling, and token sampling.
- Explicit failure-mode analysis and computational budget.

**C. Short literature appendix**

- Verify the specific near misses (especially Hebbian SAE) from full papers or code.
- State exactly why each fails one or more of the defining experimental conditions.
- Update the novelty assessment conservatively.

## 8. Recommended workflow and immediate next action

First, **confirm the mathematical positive control and an exact one-to-one matching implementation on a modest token subset**. Then test whether naive random-start Top1 training can give each token its own winning feature. Only after this succeeds—or exhibits well-characterized failure—scale to all 50,257 tokens and add rescue interventions. Preserve baseline runs even if they perform poorly: a failure to converge is scientifically meaningful precisely because the optimum is known.

**Do not change the central experiment without explicitly flagging it:** training on contextual activations, using `k > 1`, using `m < V`, permitting decoder bias, or probing embeddings with an SAE trained elsewhere would each turn this into a different study.

---

## 9. Previous project artifacts and provenance

Earlier in this conversation the following project notes were created; they can provide more detail if accessible:

- `topk_k1_dictionary_recovery_experiment.md` — brief high-level experiment plan.
- `topk_k1_token_embeddings_literature_review.md` — mathematical setup, literature background, and proposed protocol.
- `SAE_Literature_Review.md` and `sae_token_embeddings_literature_review.md` — broader related-work notes.

This handoff consolidates the points needed for an incoming agent and corrects a recurring danger: **confusing conceptual similarity with actual reproduction of the exact experiment**. The original discussion contained a focused request to search for *the exact prior experiment* and no confirmed match emerged.

**Communication / artifact preference:** The user reports that ordinary sandbox download links often fail in their ChatGPT client. When delivering files, always display the **literal `sandbox:/mnt/data/...` path as copyable text**, ideally with a Copy button; do not depend solely on a clickable download link. The user also prefers standalone, unindented LaTeX display equations rather than display math nested inside lists.

**End of handoff.**
