# Sparse Autoencoders and Token Embeddings: A Focused Literature Review

*Literature reviewed: 14 October 2026*

## Executive summary

Sparse autoencoders (SAEs) seek to reconstruct vectors as sparse combinations of learned *dictionary elements* (feature directions). In mechanistic interpretability, they are often trained on the intermediate activations of language models, in the hope that the learned directions will be more meaningful than individual neurons.

This review concerns a narrower question: **What happens if an SAE is trained directly on a language model's fixed token embedding vectors? In particular, if the SAE has exactly as many dictionary elements as the model has vocabulary entries, will it learn the original token embeddings?**

This question sits within a substantial existing literature. Earlier work demonstrated that autoencoders and sparse coding can turn *static word embeddings* into interpretable sparse representations. Later work used SAEs to analyse *contextual token activations* and transformer residual streams. Most recently, a 2026 study trained vocabulary-sized SAEs and investigated their alignment with token embeddings, although its inputs were contextual residual-stream states rather than the raw embedding table.

**Bottom line:** The exact proposed experiment was not identified in the literature reviewed here. But it should be framed as a **small, controlled study of dictionary recovery and feature identifiability**, not as the first application of SAEs to embedding spaces. The absence of an identified study is not evidence that none exists.

## 1. Background: the mathematical question

Let a language model have a vocabulary of size $V$ and a token embedding matrix

$$
E \in \mathbb{R}^{V\times d},
$$

whose rows are embeddings $e_1,\ldots,e_V\in\mathbb{R}^d$.

A typical SAE encodes $x\in\mathbb{R}^d$ into sparse coefficients $z(x)\in\mathbb{R}^m$ and reconstructs it using learned decoder directions $v_1,\ldots,v_m$:

$$
\hat x=b+\sum_{j=1}^m z_j(x)v_j.
$$

Its loss balances reconstruction accuracy against sparsity (for example, an $L^1$ penalty or a TopK constraint). The number $m$ is the **dictionary size**; it need not equal either $d$ or $V$.

If we set $m=V$, the original embeddings provide an *obvious candidate decoder dictionary*: take $v_j=e_j$ and encode input $e_i$ by the $i$th one-hot vector. This gives exact, one-sparse reconstructions in an idealized encoder–decoder model. **But the existence of this decomposition does not imply that a standard, restricted neural SAE encoder can implement it, nor that training will find it or prefer it.** In particular, a ReLU-affine encoder, decoder biases, feature normalization, frequency-weighted loss, and the chosen sparsity penalty affect the optimization problem.

There are three distinct questions to keep separate:

1. **Representability:** Can the architecture express a one-token-per-feature reconstruction?
2. **Optimization:** Does training from an unaligned initialization recover this solution?
3. **Identifiability:** Is it preferred or unique, or do shared directions and alternative sparse dictionaries fit as well or better?

The proposed experiment primarily investigates the second and third questions under clearly specified choices for the first.

## 2. Where the work fits historically

Two partly overlapping lines of research converge on this question:

- **Sparse representations of words.** Starting with pretrained word vectors such as GloVe and word2vec, researchers applied sparse coding or autoencoders to find sparse, interpretable coordinates. Here the input may genuinely be a *static* embedding for each word.
- **Mechanistic interpretability of neural networks.** Work on superposition motivated fitting learned sparse dictionaries to contextual activations inside transformers, where a token's representation changes with its context and layer.

The recent vocabulary-alignment literature brings the two strands closer together: it asks how learned directions inside a transformer relate geometrically to entries in its original token vocabulary.

### At-a-glance comparison

| Study | Input to sparse representation method | Dictionary / feature choice | Main relevance | Does it test raw embeddings with $m=V$? |
|---|---|---|---|---|
| **SPINE** (2018) | Static GloVe / word2vec word embeddings | 1,000 latent dimensions in reported experiments; 15,000 training words | Direct precedent for sparse autoencoders on static embeddings | **No** |
| **Word Equations** (2020) | Pretrained static word embeddings | Sparse coding with directions constructed from word embeddings | Demonstrates an embedding-derived dictionary, but does **not** learn it freely | **No** |
| **Conjecture interim report** (2023) | Synthetic features; separately, LM activations | Synthetic dictionary often matches number of ground-truth features | Direct precedent for testing dictionary recovery when sizes match | **No** |
| **Beyond Label Attention** (2024) | Medical RoBERTa token activations sampled from clinical notes | 6,144 learned SAE features | Contextual token representations, shared medical concepts, downstream ablations | **No** |
| **Layer-0 positional SAE note** (2024) | GPT-2 layer-0 residual stream: token **plus positional** embeddings | SAE on residual stream | Highlights why even early transformer activations differ from raw token embeddings | **No** |
| **VASAE** (2026) | Contextual post-residual-stream activations | **Exactly vocabulary-sized** dictionaries, including a conventional SAE baseline | Closest match on $m=V$ and direct tests of vocabulary alignment | **No** |

The distinction between **static token embeddings** and **context-dependent activations** is essential; authors may use the phrase “token embeddings” for both.

## 3. Closest prior studies

### 3.1 SPINE: SParse Interpretable Neural Embeddings (2018)

**Anant Subramanian, Danish Pruthi, Harsh Jhamtani, Taylor Berg-Kirkpatrick, and Eduard Hovy.** *AAAI 2018.*  
[Paper / abstract](https://arxiv.org/abs/1711.08792) · [Code](https://github.com/harsh19/SPINE)

**Research question.** Can dense word vectors be recoded into sparse, more human-interpretable dimensions without sacrificing their usefulness?

**Method.** SPINE introduces a denoising sparse-autoencoder variant with explicit sparsity objectives. The authors train on pretrained 300-dimensional GloVe and word2vec vectors. Their reported setup uses 15,000 training words and **1,000 latent dimensions**.

**Finding.** The resulting sparse dimensions can group related words; human interpretability tests favour them over the original dense coordinates, with competitive downstream performance.

**Connection to the proposed project.** This is a direct precedent for applying an autoencoder to a *finite set of fixed word embeddings*. It also suggests a plausible alternative to recovering individual words: the dictionary may capture components shared by many words. But the reported experiment does not match the number of dictionary elements to the vocabulary size, nor ask whether decoder columns reproduce the original vectors.

### 3.2 Word Equations: Inherently Interpretable Sparse Word Embeddings through Sparse Coding (2020)

**Adly Templeton.** *Preprint, 2020.*  
[Paper](https://arxiv.org/abs/2004.13847)

**Research question.** Can a sparse representation be inherently interpretable because each coordinate is tied to a familiar word or grammatical concept?

**Method.** The paper uses sparse coding with a dictionary built from existing word-embedding directions. Thus coordinates already have word-based interpretations by construction.

**Finding.** Sparse word representations based on this kind of dictionary can retain useful performance while exposing more understandable dimensions.

**Connection.** The key methodological distinction is **fixed versus learned dictionaries**. Constructing a dictionary from word vectors demonstrates the feasibility of word-aligned sparse representations, but not whether an unconstrained SAE *discovers* that dictionary from data. It is a useful baseline for the proposed experiment.

### 3.3 Conjecture: Taking Features Out of Superposition with Sparse Autoencoders (2023)

**Lee Sharkey, Beren Millidge, and Dan Braun.** *Interim research report, 2023.*  
[Report](https://www.conjecture.dev/research/interim-research-report-taking-features-out-of-superposition-with-sparse-autoencoders)

**Research question.** Can a sparse autoencoder recover known generating directions hidden by superposition?

**Method.** On synthetic data, the researchers generate activations from a known set of feature directions and train an SAE to infer them. In the principal toy setup, **512 learned dictionary elements match 512 generating features**. Recovery is quantified using cosine similarity to the known directions. The report also examines real language-model activations, where the problem is less straightforward.

**Finding.** Good recovery is possible on appropriate synthetic data and hyperparameter settings; low reconstruction error alone is insufficient to establish recovery.

**Connection.** This provides an especially useful **methodological blueprint**: distinguish reconstruction from ground-truth direction recovery, sweep hyperparameters, and watch for duplicate or inactive features. Here, the token embedding matrix supplies a known *candidate* dictionary, but unlike synthetic data it is not necessarily the unique latent generating dictionary.

### 3.4 Beyond Label Attention: Transparency in Language Models for Automated Medical Coding via Dictionary Learning (2024)

**John Wu, David Wu, and Jimeng Sun.** *EMNLP 2024.*  
[Paper](https://aclanthology.org/2024.emnlp-main.500/) · [Readable arXiv version](https://arxiv.org/html/2411.00173)

**Research question.** Can sparse dictionary features make clinical language-model predictions more interpretable than token-level attention alone?

**Method.** The authors fit $L^1$-regularized and SPINE-style SAEs to **token representations generated by a medical RoBERTa model on clinical notes**, using **6,144 dictionary features**. These are sampled, context-bearing model activations—not a one-pass enumeration of the fixed vocabulary embedding matrix.

**Finding.** Features can group medically related occurrences and support ablation-based explanations of predicted diagnosis codes. For example, a feature can activate on different occurrences associated with depression-related concepts.

**Connection.** The paper establishes that sparse decomposition of token-level model representations need not produce one feature per token. However, its contextual inputs and smaller dictionary mean it cannot settle the narrower static-embedding/$m=V$ conjecture.

### 3.5 Understanding Positional Features in Layer 0 SAEs (2024)

**Bilal Chughtai and Yeu-Tong Lau.** *Informal research note, July 2024.*  
[Research note](https://bilalchughtai.co.uk/pos-sae/)

**Research question.** What positional structure is learned by SAEs trained on GPT-2's earliest residual-stream activations?

**Method and finding.** The SAE is trained on the layer-0 residual stream, which contains both token and positional embedding contributions. Its learned positional directions collectively describe structure in the positional embedding manifold.

**Connection.** This is a reminder that **“layer 0” does not automatically mean “the raw token embedding table.”** Position contributions change the data distribution even before attention and MLP computations begin.

### 3.6 VASAE: Naming SAE Dictionary Directions with Vocabulary-Aligned Anchoring (2026)

**Kairui Zhang, Ziwen Yu, Zahraa S. Abdallah, and Martha Lewis.** *2026 preprint; accepted to the 2nd Workshop on Compositional Learning at ICML 2026.*  
[Paper](https://arxiv.org/html/2606.27941) · [Project page](https://karry-z.github.io/VASAE/)

**Research question.** Can SAE features learned from transformer residual streams be assigned vocabulary-based names by encouraging their decoder directions to lie near actual token-embedding directions?

**Method.** The authors compare three models on **contextual post-residual-stream activations**: an ordinary learned SAE, a variant whose decoder is **fixed to the token embeddings**, and their proposed SAE with an additional **soft vocabulary-alignment penalty**. All three use vocabulary-sized dictionaries and a TopK sparsity constraint ($k=32$). Model sizes include GPT-2 Small ($V=50{,}257$, $d=768$) and Llama 3.1 8B ($V=128{,}256$, $d=4{,}096$).

**Findings.** For GPT-2 Small, ordinary SAE decoder directions show relatively low nearest-token cosine similarities (reported concentrations around $0.1$–$0.2$). Under vocabulary anchoring, approximately 89–94% of features in layers 0–10 exceed cosine similarity $0.8$ with some token embedding; importantly, **this is not a one-to-one matching**. The **hard-tied decoder** performs substantially worse at reconstructing the contextual residual streams than a freely learned dictionary.

**Connection.** This is the closest paper to the proposed project, and it already answers a related question: **having $m=V$ by itself does not make a standard SAE on contextual residual streams recover token-embedding directions**. What it does *not* test is whether recovery occurs when the SAE's entire training dataset is precisely the set of *raw* token embeddings. The hard-tied decoder result should not be transferred to that different dataset without an experiment.

## 4. The remaining research question

The specifically proposed study would:

1. Extract only the model's **input token embedding matrix** $E$ (no positional encodings, no transformer forward pass, no contextual residual states).
2. Train an SAE whose learned decoder contains **$m=V$ features**, initially without vocabulary-alignment supervision.
3. Ask whether its learned dictionary approximates the rows of $E$, up to permutation, scaling, and normalization choices.
4. Determine whether it instead recovers shared semantic, syntactic, or geometric components across tokens.

The hypothesis is **not** that every embedding must define a true atomic concept. Rather, it is that one-token-per-feature is an obvious candidate factorization, and so the conditions under which training does or does not find it may illuminate the inductive biases of sparse dictionary learning.

This modest question connects to the larger programme of understanding **feature recovery, superposition, dictionary identifiability, and the relationship between vocabulary geometry and internal LLM representations**.

## 5. Minimal experimental design suggested by the literature

**Initial model.** GPT-2 Small is a tractable choice: $V=50{,}257$, $d=768$. Train on its embedding rows, optionally sampled according to a controlled token distribution. Start with $m=V$ and compare smaller and larger dictionaries as resources permit.

**Essential comparisons**

- **Learned SAE:** unanchored decoder, with $L^1$ sparsity and/or TopK variants.
- **Embedding-tied decoder:** fix the decoder directions to $E$ and train only the encoder. This controls for architectural constraints on recovering the trivial dictionary.
- **Soft vocabulary anchoring:** an optional comparison informed by VASAE—not the default, because it biases the very result under investigation.
- **Simpler sparse coding:** optimize sparse coefficients directly for each embedding; this separates encoder capacity from dictionary capacity.

**Evaluate more than reconstruction error**

- **Reconstruction and sparsity:** MSE, relative reconstruction error, average active features ($L_0$), and coefficient sizes.
- **Direction recovery:** nearest-embedding cosine similarity in both directions (learned feature $\to$ token and token $\to$ learned feature).
- **One-to-one recovery:** global matching or approximate assignment, with a distribution of matched cosine similarities. Nearest-neighbour scores alone can be misleading because several features may match the same token.
- **Feature reuse:** how many embeddings activate each feature, and how many features each embedding activates.
- **Robustness:** multiple random seeds, sparsity strengths, feature counts, normalized versus raw inputs, and uniform versus frequency-weighted token sampling.

A meaningful result could be *either* recovery or non-recovery, provided the experiment rules out simple failure modes (poor optimization, dead features, or a restrictive encoder that cannot implement a near-one-hot code).

## 6. How to position the contribution

A suitable short description is:

> **A controlled empirical study of vocabulary recovery in sparse autoencoders.** We investigate whether unsupervised SAEs trained directly on a language model's fixed token embedding matrix recover the original token embeddings when the learned dictionary is vocabulary-sized. We compare reconstruction, sparse feature reuse, and geometric dictionary recovery, relating the results to earlier sparse word representations and modern vocabulary-aligned SAE research.

This would be a **targeted benchmark**, not a claim to introduce SAEs on embeddings, discover token-related SAE features for the first time, or resolve the general identifiability problem. Its distinctive contribution is to **isolate static embedding inputs and exactly vocabulary-sized dictionaries** in a setting where candidate target directions are known.

## 7. Research limitations and reading notes

- **Literature coverage is not exhaustive.** These are the closest papers identified in the searches underlying this review, not proof that an exact-match experiment has never been published.
- **Static and contextual embeddings are often called the same thing.** Source methodology matters more than the terminology used in an abstract.
- **Vocabulary-sized does not mean vocabulary-aligned.** $m=V$ gives the dictionary enough slots but imposes no token correspondence.
- **Alignment is not identity or causal interpretation.** High cosine similarity to a token embedding is geometric evidence, not proof of a feature's exclusive semantic meaning.
- **The ideal one-hot solution depends on the function class.** An arbitrary lookup encoder can implement it; a simple affine-ReLU encoder need not.

## References and suggested reading order

**For a cold reader:** Begin with SPINE for static embedding decompositions, then the Conjecture report for recovery metrics, then VASAE for the closest modern experiment. Read Beyond Label Attention and Word Equations for alternative motivations and baselines.

1. Subramanian, A., Pruthi, D., Jhamtani, H., Berg-Kirkpatrick, T., & Hovy, E. (2018). **SPINE: SParse Interpretable Neural Embeddings.** AAAI. https://arxiv.org/abs/1711.08792
2. Templeton, A. (2020). **Word Equations: Inherently Interpretable Sparse Word Embeddings through Sparse Coding.** https://arxiv.org/abs/2004.13847
3. Sharkey, L., Millidge, B., & Braun, D. (2023). **Taking Features Out of Superposition with Sparse Autoencoders.** Interim research report. https://www.conjecture.dev/research/interim-research-report-taking-features-out-of-superposition-with-sparse-autoencoders
4. Wu, J., Wu, D., & Sun, J. (2024). **Beyond Label Attention: Transparency in Language Models for Automated Medical Coding via Dictionary Learning.** EMNLP. https://aclanthology.org/2024.emnlp-main.500/
5. Chughtai, B., & Lau, Y.-T. (2024). **Understanding Positional Features in Layer 0 SAEs.** Research note. https://bilalchughtai.co.uk/pos-sae/
6. Zhang, K., Yu, Z., Abdallah, Z. S., & Lewis, M. (2026). **VASAE: Naming SAE Dictionary Directions with Vocabulary-Aligned Anchoring.** https://arxiv.org/abs/2606.27941

### Broader context (optional)

- Elhage, N., et al. (2022). [**Toy Models of Superposition**](https://transformer-circuits.pub/2022/toy_model/index.html). Explains why a network may represent more features than individual neuron dimensions.
- Bricken, T., et al. (2023). [**Towards Monosemanticity: Decomposing Language Models With Dictionary Learning**](https://www.transformer-circuits.pub/2023/monosemantic-features/index.html). A major demonstration of SAEs as tools for mechanistic interpretability.
