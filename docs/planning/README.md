# Planning documents

These documents come from the planning discussion that preceded the code. They are kept unchanged as a record. Where they differ from the current design, the current design is in the experiment plans in `docs/` and the repository README.

- `SAE_Experiment_1.md`: the original specification of the first experiment.
- `SAE_Project_Handoff.md`: the handoff from the planning discussion.
- `SAE_Lit_Review_1.md` and `SAE_Lit_Review_2.md`: literature reviews.

Changes to `SAE_Experiment_1.md` agreed since it was written:

- Only the random-start SAE is run. The embedding-initialised control and the classical dictionary-learning baseline are dropped for now. There is one seed, so stability across seeds is not measured.
- Tokens are sampled uniformly. Frequency weighting exists as a toggle but is off.
- The experiment was split up. Experiment 1 measured how many Top1 features that are dead at initialisation revive (none did). Experiment 2 is an L1-penalised ReLU SAE. Experiment 3 is a TopK SAE whose $k$ is lowered from 32 to 1 during training.
