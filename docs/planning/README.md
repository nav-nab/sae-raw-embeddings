# Planning documents

These documents come from the planning discussion that preceded the code. They are kept unchanged as a record. Where they differ from the current design, the current design is in `docs/experiment_1_top1.md`, `docs/experiment_2_l1_relu.md` and the repository README.

- `SAE_Experiment_1.md`: the original specification of the first experiment.
- `SAE_Project_Handoff.md`: the handoff from the planning discussion.
- `SAE_Lit_Review_1.md` and `SAE_Lit_Review_2.md`: literature reviews.

Changes to `SAE_Experiment_1.md` agreed since it was written:

- Only the random-start SAE is run. The embedding-initialised control and the classical dictionary-learning baseline are dropped for now. There is one seed, so stability across seeds is not measured.
- Tokens are sampled uniformly. Frequency weighting exists as a toggle but is off.
- The experiment was split in two. Experiment 1 measures how many Top1 features that are dead at initialisation revive. Experiment 2, an L1-penalised ReLU SAE, runs if fewer than 80% revive.
