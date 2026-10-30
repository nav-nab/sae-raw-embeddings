# Write-up outline

A streamlined plan of the write-up, organised as a fleshed-out table of contents. Each subsection makes one conceptual point and names the experiments that support it. The full details, numbers and tables are in `docs/write_up_plan.md` (row tags such as T21 and R1 refer to its tables) and in the per-experiment docs. The final write-up will sit between the two in detail.

## Thesis

### A toy model with a known answer

SAEs are usually trained on model activations. There, the true features are unknown and far outnumber the dimensions. So when an SAE fails to find a clean feature, it is hard to tell why. The training may have gone wrong, or the dictionary may be too small, or the feature may not exist at all.

We study a setting where the answer is known. The data are the rows of GPT-2 Small's token embedding table: $V = 50{,}257$ vectors in $d = 768$ dimensions. We give the SAE exactly as many features as there are tokens, $m = V$.

With $m = V$ there is no bottleneck. Every token can have a feature of its own. A one-hot code, with each token rebuilt by a single feature pointing along it, reconstructs every token exactly and is as sparse as a code can be. The natural target for the SAE is therefore the embedding table itself: one feature per token, along that token's embedding. Sections 1.1 and 2.1 make this precise for each architecture.

This makes $m = V$ a toy model of SAE training. Because the target is known, we can measure how much of it training finds: a token is recovered if some feature matches its embedding. And because capacity is not the limit, any token that is not recovered is a failure of training. We train from random initialisation, as in practice, and ask which architectures and settings find the target, and at what cost.

### The findings

Training can find the target with either architecture, in the right regime. The architecture decides which regimes work and how much training they cost.

- **TopK SAEs** recover the embeddings only when $k$ is annealed from a large value down to 1, and at a high training cost (Section 1).
- **ReLU SAEs with an L1 penalty** recover them across a wide region of hyperparameters (Section 2).
- **With training scaled to the vocabulary, recovery does not depend on its size**, for both architectures. A 2,000-token subvocabulary is therefore a faithful small-scale model of the full table, and the full-scale runs bear this out (Section 3).

## Setting and methods

- **Data.** The GPT-2 Small input embedding table ($V = 50{,}257$, $d = 768$), and subvocabularies of its most frequent tokens: the 2k subset, and 5,000, 10,000 and 20,000 tokens.
- **Model.** $m = V$ features, unit-norm decoder rows, no decoder bias. Random initialisation, with the encoder taken to be the transpose of the decoder.
- **Training.** Uniform sampling of tokens, Adam at a constant learning rate scaled as $1/\sqrt{m}$ (the "Gao rate").
- **Measurement.** A token is recovered if some decoder row is its mutual nearest neighbour at cosine above 0.9 (and 0.99). Run-to-run noise is about ±20 tokens on the 2k subset.
- The training setup of every run is tabulated in Appendix C.

## 1. TopK SAEs: recovery by annealing $k$, at a high training cost

### 1.1 The target

For Top1 ($k = 1$), the zero-loss dictionary is exactly the set of normalised embeddings. For $k > 1$, tokens can be rebuilt from several features, so this uniqueness is lost. *(Theory.)*

### 1.2 Sparsity against keeping features alive

A TopK feature gets gradient only when it is selected. Top1 from the start leaves most features dead, and the few survivors mostly fail to match a token. A larger fixed $k$ keeps features alive, but codes become blends of $k$ features. Neither recovers the embeddings. *(Experiment 1; the fixed-$k$ runs on the 2k subset.)*

### 1.3 Annealing: features die only at the drops in $k$, and the survivors are correct

Annealing starts at a large $k$ and lowers it in stages to 1. Features die at each drop and almost never revive. A feature survives the drop to $k = 1$ only if it is the top feature of some token, which is a competition for a limited number of slots. This count predicts the survivors to within about 2%. Unlike Top1 from the start, 96% to 100% of the survivors are recovered tokens, so recovery is essentially the number of survivors. *(Experiments 3, 7 and 8.)*

### 1.4 An auxiliary loss cannot rescue the dead features

AuxK trains dead features to reconstruct the residual. This addresses a lack of gradient, but the problem here is competition for slots. A dead feature aligned with a token's residual scores about $\lvert x\rvert\sin\theta$ on it, against $\lvert x\rvert\cos\theta$ for the feature that already holds it, so it does not win the token. In practice AuxK gave no clear gain. *(AuxK runs on the 2k subset.)*

### 1.5 Two levers: the starting $k$ and the amount of training

- **The starting $k$.** Recovery rises with $k_0$ until about 93% of features are alive at the start, then levels off.
- **The amount of training.** Recovery rises by about 3 percentage points per doubling of training, from 75% to 89% at 128,000 steps, with no sign of a plateau. Longer runs would likely recover more. Lengthening the $k = 2$ stage alone is not the lever: its contribution levels off within the stage.

*(The 2k-subset schedule grid; Experiment 7, rows T14 to T21.)*

### 1.6 The cost against ReLU

On the 2k subset, the best TopK run recovered 89% from about 66 million training tokens. The best ReLU run recovered 98% from about 8 million. *(Experiments 4 and 7.)*

## 2. ReLU SAEs with an L1 penalty: recovery across a wide region

### 2.1 The embeddings minimise the L1 loss

No code reconstructs a token at lower loss than a single feature pointing along it, with coefficient $\lvert x\rvert - \lambda/2$. With $m = V$, every token can have such a feature, so the normalised embeddings are the minimiser. The proof is a short bound from the triangle inequality. *(Appendix A.)*

### 2.2 One-hot codes are reachable for every GPT-2 token

A ReLU feature can fire on one token alone exactly when that token's embedding is a vertex of the convex hull of the table. Every GPT-2 embedding is a vertex. How widely a token is separated from the others predicts whether training recovers it: the tokens that are missed are those separated only by a narrow gap. *(Appendix B; Experiment 6, with recovery from Experiment 2.)*

### 2.3 Three regimes, and codes that end near one-hot

- **Three regimes.** With too little training, codes stay dense. With too much pressure (a large $\lambda$ or learning rate), features die early. In between lies a wide region where recovery exceeds 80%. Progress depends on learning rate × steps, not on the batch size.
- **Gradual sparsification.** Codes sparsify one token at a time, unlike TopK's sudden drops, and end near one-hot. Most tokens use a single feature, with a short tail using more.
- **A cutoff on small coefficients changes little.** Ignoring features whose coefficient is below 10% of the token's largest (those contributing about 1% of the reconstruction or less) moves the share of tokens on one feature by at most 2 points.

*(Experiment 4, rows R1 to R4.)*

### 2.4 The penalty must be weak, but not infinitely weak

A smaller $\lambda$ recovers more, because it kills fewer features. But the time to sparsify grows as $\lambda$ falls, roughly as $\lambda^{-0.7}$. At $\lambda = 0$ the embeddings are no longer singled out by the loss, and training recovers nothing: codes stay dense. So the best $\lambda$ is set by the training budget. It is roughly the smallest $\lambda$ that still finishes sparsifying in the time available. *(Experiments 4 and 10.)*

## 3. With training scaled to the vocabulary, recovery does not depend on its size

### 3.1 Measuring training across sizes

At a fixed vocabulary size, learning rate × steps is what counts. Across sizes, the steps needed grow in proportion to $V$. *(Experiments 4, 7 and 9.)*

### 3.2 ReLU: recovery as a function of $\lambda$ is the same at every size

With steps in proportion to $V$, each $\lambda$ recovers within a few points of the same share at 2,000, 5,000, 10,000 and 20,000 tokens, and the best $\lambda$ does not change. *(Experiment 9, rows R6 to R9.)*

### 3.3 TopK: match the share alive at the start, and recovery is the same at every size

Starting from the $k_0$ that leaves about 93% of features alive, with steps in proportion to $V$, annealing recovers 74% to 77% at 5,000, 10,000 and 20,000 tokens, as on the 2k subset. The starting value follows a power law, $k_0 \approx 1.37\,V^{0.405}$. This gives a recipe for carrying the schedule to any size. *(Experiments 5 and 8, rows T22 to T24.)*

### 3.4 The role of geometry

The law for $k_0$ depends on which tokens make up the vocabulary. Rarer tokens crowd more closely around a shared direction, and crowded directions leave fewer features alive at the start. For random subsets of tokens, which keep the same geometry at every size, the law is a cleaner power law ($k_0 \approx 2.90\,V^{0.345}$). The separation margins of Section 2.2 barely change with size, so differences in recovery across sizes come from training, not from tokens becoming harder to isolate. *(Experiments 5 and 6.)*

### 3.5 Full scale: the 2k subset is a faithful small-scale model

On the full table, ReLU with $\lambda = 0.3$ recovered 93%, and TopK annealing recovered 70%. Both fit what the smaller vocabularies predict for the training they received. *(Experiments 2 and 3.)*

## Appendices

- **A. Why the embeddings minimise the L1 loss:** the step-by-step argument.
- **B. When a ReLU encoder can output one-hot codes:** the convex-hull condition, and the check for GPT-2.
- **C. Training setup:** settings common to every run, and a table of each experiment's model, vocabulary, batch size, steps, learning rate and training length.
