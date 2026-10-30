# Documentation

The paper (`paper/main.pdf`) is the final account of the project. The documents here are the working notes written as the project went: each experiment's plan, how to run it, and its results and interpretation at the time. Later experiments sometimes revised earlier readings; where they did, the later document says so, and the paper gives the settled view.

## Experiments

| Experiment | Document | What it tests | Paper |
|---|---|---|---|
| 1 | `experiment_1_top1.md`, `experiment_1_results.md` | Top1 SAE on the full table: do features dead at initialisation revive? | Section 3.2 |
| 2 | `experiment_2_l1_relu.md` | L1 ReLU SAE on the full table ($\lambda = 0.3$) | Sections 4.1, 4.3, 5.5 |
| 3 | `experiment_3_k_annealing.md` | TopK with $k$ annealed to 1 on the full table; also the 2k-subset pilots (fixed $k$, schedules, AuxK) | Sections 3.2 to 3.5, 5.5 |
| 4 | `experiment_4_relu_grid.md` | Grid of L1 ReLU SAEs on the 2k subset ($\lambda$, learning rate, batch size) | Sections 4.3, 4.4 |
| 5 | `experiment_5_alive_at_init.md` | Features alive at initialisation, by vocabulary size and $k$ | Sections 5.3, 5.4 |
| 6 | `experiment_6_hull_vertices.md` | Which embeddings a ReLU encoder can code one-hot (convex-hull vertices) | Section 4.2, Appendix B |
| 7 | `experiment_7_topk_lr_steps.md` | TopK annealing with more training and other learning rates | Sections 3.3, 3.5, 3.6 |
| 8 | `experiment_8_topk_sizes.md` | TopK annealing at 5,000, 10,000 and 20,000 tokens | Section 5.3 |
| 9 | `experiment_9_relu_sizes.md` | L1 ReLU SAEs at 5,000, 10,000 and 20,000 tokens | Section 5.2 |
| 10 | `experiment_10_relu_no_penalty.md` | Control: ReLU SAE with no penalty ($\lambda = 0$) | Section 4.4 |

## Other documents

- `failure_modes.md`: a running table of the attempts that failed to recover the embeddings, with an interpretation of each.
- `write_up_plan.md` and `write_up_outline.md`: the plans for the paper. The full plan has every results table, with row tags (T1 to T24 for TopK, R1 to R10 for ReLU) that some experiment documents refer to. The outline is the paper's structure.
- `planning/`: the documents from the planning discussion before any code was written, kept unchanged as a record.

## Notes on the record

- Several documents mention model weights, checkpoints and snapshots under `runs/`. These were deleted when the repository was published, except the five final models read by `scripts/code_spread.py` and Experiment 6, which are kept locally but not tracked by git. The result tables (CSV and JSON) under `runs/` are tracked.
- Times quoted as "sandbox" or "laptop" refer to the two machines used: a cloud sandbox for short pilots, and a laptop (Intel CPU, PyTorch 2.2.2) for the long runs.
- The schedule names in `experiment_3_k_annealing.md` (A to P) are those of the pilot grid. The schedule used from then on, "O", is the one the paper calls the schedule with a long $k = 2$ stage: 128, 64, 32, 16, 8 and 4 share the first 40% of training, then $k = 2$ for 40% and $k = 1$ for 20%.
