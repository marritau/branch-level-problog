# Changes

## Repository Layout Cleanup

- Reorganized the repository so tests, diagnostics, and generated artifacts are no longer mixed in the project root.
- Branch-probability tests now live under `tests/branch/`.
- ProbLog-related tests now live under `tests/problog/`.
- E2E-related tests now live under `tests/e2e/`.
- Diagnostic scripts now live under `scripts/diagnostics/`.
- Generated files are now separated into:
  - `output/checkpoints/`
  - `output/problog/`
  - `output/debug_export/`
  - `output/benchmarks/`
- Added `project_paths.py` to centralize shared repository paths and reduce hard-coded output locations.
- Updated training and diagnostic code to use the new structured paths without breaking the existing tests and export checks.

## Branch-to-Class ProbLog Support Rules

- Added fixed ProbLog class-support rules to the latent export.
- Each branch now exports rules of the form `theta::supports(branch,class,X) :- z(branch,X).`
- The `theta` values are initialized from `Branch.class_proportions`, preserving the original BranchNet interpretation of the frozen `W2` head at the parent-of-leaf branch level.
- The new rules keep the branch id in `supports(...)` so multiple active, including nested, parent-of-leaf branches can contribute separately to class support.
- Added a minimal class predicate of the form `class(X,class) :- supports(Branch,class,X).` to aggregate support from all active branches.
- Added optional `query(class(object,class)).` generation for each exported object and class via `include_class_queries=True`, enabling ProbLog class-probability queries without neural softmax when class-level inference is requested.
- Enabled class-query generation in the training-time latent export, so `Trainer.train(..., branch_problog_path=...)` writes a class-query-ready ProbLog file.
- Trainable `theta` parameters are not implemented in this stage; this is the fixed class-support head initialized from existing BranchNet class proportions.

## Differentiable Branch Prior Access

- Added `predict_branch_proba_torch(...)` to `BranchNetFramwork.py` as a differentiable mirror of `predict_branch_proba(...)`.
- The new method returns branch-level probabilities as a `torch.Tensor` without `torch.no_grad()`, `detach()`, NumPy conversion, or forced CPU transfer.
- This preserves the computation graph and opens the path for end-to-end optimization of downstream ProbLog-inspired class heads.
- Added a dedicated test that verifies:
  - the returned value is a tensor,
  - `requires_grad=True`,
  - gradients flow back into `W1` through `backward()`.

## Differentiable ProbLog Class Head

- Added `differentiable_problog.py` with `DifferentiableClassHead`, a PyTorch noisy-or class head aligned with the existing ProbLog rules.
- The head stores a trainable parameter matrix `theta` of shape `[n_branches, n_classes]`, initialized from `Branch.class_proportions`.
- Internally, `theta` is parameterized in logit space and mapped back to probabilities with `sigmoid`, preserving the independent `[0,1]` semantics of `supports(branch,class,X)`.
- The `forward(z)` method expects branch priors `z = P(z_b | x)` and computes class probabilities with:
  - `P(class=c | x) = 1 - Π_b (1 - theta_bc * z_b)`
- The noisy-or computation is implemented with `log1p(-support)` for numerical stability, avoiding underflow when the number of branches becomes large.
- Added an end-to-end unit test that verifies:
  - correct initialization from `class_proportions`,
  - exact noisy-or values on a small example,
  - valid probability range,
  - gradient flow to both the branch-prior input and trainable `theta`.

## End-to-End Training Loop

- Added `train_e2e.py` with a baseline end-to-end training loop for the differentiable noisy-or class head.
- The training loop assumes a ready `BranchNetModel` already built from an `ExtraTrees` ensemble and trains a `DifferentiableClassHead` on top of branch priors `P(z_b | x)`.
- Added an optional `train_w1=True/False` flag:
  - `False` trains only the class-rule parameters `theta`,
  - `True` jointly trains `theta` and BranchNet `W1`.
- Implemented two loss modes:
  - `loss_mode='nll'`: normalizes noisy-or class beliefs into a distribution before `NLLLoss`,
  - `loss_mode='bce'`: treats class outputs as one-vs-rest probabilities and uses binary cross-entropy against one-hot targets.
- The `BCE` mode is the more natural fit for the current noisy-or semantics, while the `NLL` mode is kept as a requested multiclass baseline.
- Added helper inference utilities for e2e class probabilities and predictions.
- Added `assign_theta_to_branches(...)` to write trained `theta` values back into `branch.class_proportions` for later ProbLog export.
- Added smoke tests that run both loss modes on a small Iris-based example and verify stable outputs.

## Native ProbLog Round-Trip Consistency

- Added `tests/e2e/test_e2e_problog_consistency.py` for the prior-based noisy-or baseline.
- The test:
  - trains the differentiable class head in `BCE` mode,
  - writes the trained `theta` values back into exported branches,
  - regenerates a latent `.pl` file with `include_class_queries=True`,
  - runs that program through native ProbLog,
  - compares native ProbLog class probabilities with the PyTorch noisy-or forward pass.
- The round-trip comparison is performed without `observed_data`, so both sides use the same branch-prior semantics `P(z_b | x)`.
- This keeps the `.pl <-> PyTorch` consistency invariant intact for the baseline e2e pipeline before a differentiable posterior module is introduced.

## Benchmark Driver

- Added `compare_e2e.py` as a benchmark driver over built-in scikit-learn datasets:
  - `iris`
  - `wine`
  - `breast_cancer`
  - `digits`
- The main benchmark compares:
  - `ExtraTrees`
  - `BranchNet-Neural (frozen W2)`
  - `Rita-e2e-NoisyOr-BCE`
  - `Rita-e2e-NoisyOr-NLL`
- Added optional `train_w1=True` ablation variants for the e2e models.
- Metrics include:
  - accuracy
  - weighted F1
  - MCC
  - log loss
- For the `BCE` e2e variant, class beliefs are normalized before computing `log loss`, so that the metric is comparable to the probabilistic baselines.
- Results are written to `output/benchmarks/compare_e2e_results.txt` by default and are flushed incrementally during long runs.
- Added terminal progress reporting with dataset, fold, model, elapsed time, and ETA estimates.
- Added a smoke test `tests/e2e/test_compare_e2e_smoke.py` that runs a compact one-dataset benchmark pass on Iris.

## Initial Benchmark Observations

- A short benchmark with `branchnet_epochs=20` and `e2e_epochs=20` is fast to run, but it under-trains the e2e noisy-or models.
- Under the short schedule, the e2e variants are clearly weaker than both `ExtraTrees` and `BranchNet-Neural`, especially on `iris` and `breast_cancer`.
- Increasing the schedule to `branchnet_epochs=100` and `e2e_epochs=100` improves the e2e models substantially:
  - on `iris`, accuracy rises well above the near-random `~0.33` regime seen in the short run,
  - on `breast_cancer`, the e2e variants move away from near-majority-class behavior toward much more meaningful classification.
- Even after the longer run, `ExtraTrees` and `BranchNet-Neural (frozen W2)` remain stronger baselines than the current noisy-or e2e head.
- The `NLL` variant is usually slightly better than the `BCE` variant on the longer runs, but the gap is modest.
- Fine-tuning `W1` changes results only slightly in the current setup; it does not by itself close the gap to the stronger baselines.
- The current conclusion is that the e2e noisy-or pipeline is working and trainable, but it is still a weaker classifier than the original neural head and benefits noticeably from longer training schedules.
