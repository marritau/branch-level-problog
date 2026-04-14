# Changes

## Branch-to-Class ProbLog Support Rules

- Added fixed ProbLog class-support rules to the latent export.
- Each branch now exports rules of the form `theta::supports(branch,class,X) :- z(branch,X).`
- The `theta` values are initialized from `Branch.class_proportions`, preserving the original BranchNet interpretation of the frozen `W2` head at the parent-of-leaf branch level.
- The new rules keep the branch id in `supports(...)` so multiple active, including nested, parent-of-leaf branches can contribute separately to class support.
- Added a minimal class predicate of the form `class(X,class) :- supports(Branch,class,X).` to aggregate support from all active branches.
- Added optional `query(class(object,class)).` generation for each exported object and class via `include_class_queries=True`, enabling ProbLog class-probability queries without neural softmax when class-level inference is requested.
- Enabled class-query generation in the training-time latent export, so `Trainer.train(..., branch_problog_path=...)` writes a class-query-ready ProbLog file.
- Trainable `theta` parameters are not implemented yet; this is the fixed class-support head initialized from existing BranchNet class proportions.
