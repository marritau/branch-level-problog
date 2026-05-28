import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from compare_e2e import (
    DEFAULT_DATASETS,
    compute_metrics,
    evaluate_branchnet_neural,
    evaluate_extratrees,
    evaluate_rita_e2e,
    format_duration,
    format_metrics,
    load_builtin_dataset,
    summarize_records,
)
from project_paths import BENCHMARKS_DIR


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    head_type: str
    theta_init_mode: str
    loss_mode: str
    use_posterior: bool
    train_w1: bool
    learnable_competition: bool = False


def _recommended_loss_mode(head_type: str) -> str:
    if head_type == "noisy_or":
        return "bce"
    if head_type == "softmax_competition":
        return "nll"
    raise ValueError(
        f"Unsupported head_type: {head_type}. Expected 'noisy_or' or 'softmax_competition'."
    )


def _build_experiment_name(
    head_type: str,
    theta_init_mode: str,
    loss_mode: str,
    use_posterior: bool,
    train_w1: bool,
    learnable_competition: bool,
) -> str:
    head_label = "NoisyOr" if head_type == "noisy_or" else "SoftmaxCompetition"
    theta_label = "Weighted" if theta_init_mode == "weighted" else "NormTheta"
    parts = ["Rita", head_label, theta_label, loss_mode.upper()]
    if learnable_competition:
        parts.append("LearnableCompetition")
    if use_posterior:
        parts.append("Posterior")
    if train_w1:
        parts.append("trainW1")
    return "-".join(parts)


def build_default_ablation_specs(
    heads: Sequence[str] = ("noisy_or", "softmax_competition"),
    theta_init_modes: Sequence[str] = ("weighted", "normalized"),
    posterior_options: Sequence[bool] = (False, True),
    train_w1_options: Sequence[bool] = (False, True),
    use_learnable_competition: bool = False,
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for head_type in heads:
        for theta_init_mode in theta_init_modes:
            for use_posterior in posterior_options:
                for train_w1 in train_w1_options:
                    loss_mode = _recommended_loss_mode(head_type)
                    learnable_competition = (
                        use_learnable_competition and head_type == "softmax_competition"
                    )
                    specs.append(
                        ExperimentSpec(
                            name=_build_experiment_name(
                                head_type=head_type,
                                theta_init_mode=theta_init_mode,
                                loss_mode=loss_mode,
                                use_posterior=use_posterior,
                                train_w1=train_w1,
                                learnable_competition=learnable_competition,
                            ),
                            head_type=head_type,
                            theta_init_mode=theta_init_mode,
                            loss_mode=loss_mode,
                            use_posterior=bool(use_posterior),
                            train_w1=bool(train_w1),
                            learnable_competition=bool(learnable_competition),
                        )
                    )
    return specs


def run_ablation_matrix(
    datasets: Iterable[str] = DEFAULT_DATASETS,
    experiment_specs: Optional[Sequence[ExperimentSpec]] = None,
    folds: int = 5,
    seed: int = 42,
    include_baselines: bool = True,
    train_reliability: bool = True,
    p_high: float = 0.95,
    p_low: float = 0.05,
    branch_truth_loss_weight: float = 0.0,
    use_class_leak: bool = False,
    init_class_leak: float = 0.0,
    train_class_leak: bool = True,
    use_output_calibration: bool = False,
    init_calibration_temperature: float = 1.0,
    train_calibration: bool = True,
    theta_prune_threshold: float = 0.0,
    theta_l1_weight: float = 0.0,
    class_leak_l1_weight: float = 0.0,
    calibration_l2_weight: float = 0.0,
    branchnet_epochs: int = 20,
    branchnet_lr: float = 1e-2,
    e2e_epochs: int = 20,
    e2e_lr: float = 1e-3,
    device: str = "cpu",
    output_path: Optional[str | Path] = None,
    max_folds: Optional[int] = None,
) -> tuple[list[dict], list[dict], Path]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    specs = list(experiment_specs) if experiment_specs is not None else build_default_ablation_specs()
    if not specs:
        raise ValueError("run_ablation_matrix requires at least one ExperimentSpec")

    output_file = Path(output_path) if output_path is not None else BENCHMARKS_DIR / "ablation_matrix_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    datasets = list(datasets)
    folds_to_run = min(folds, max_folds) if max_folds is not None else folds
    models_per_fold = len(specs) + (2 if include_baselines else 0)
    total_tasks = len(datasets) * folds_to_run * models_per_fold
    completed_tasks = 0
    task_durations: list[float] = []
    benchmark_start = time.perf_counter()

    records: list[dict] = []
    lines: list[str] = []

    def flush_output() -> None:
        output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"[ablation-matrix] starting: datasets={datasets}, folds={folds_to_run}, "
        f"experiments={len(specs)}, include_baselines={include_baselines}, total_tasks={total_tasks}",
        flush=True,
    )
    lines.append(
        f"# Ablation matrix start: datasets={datasets}, folds={folds_to_run}, "
        f"experiments={len(specs)}, include_baselines={include_baselines}, total_tasks={total_tasks}"
    )
    lines.append("=== Experiment Specs ===")
    for spec in specs:
        lines.append(
            f"{spec.name}: head={spec.head_type}, theta_init={spec.theta_init_mode}, "
            f"loss={spec.loss_mode}, posterior={spec.use_posterior}, "
            f"train_w1={spec.train_w1}, learnable_competition={spec.learnable_competition}"
        )
    lines.append("")
    flush_output()

    head_kwargs = {
        "use_class_leak": use_class_leak,
        "init_class_leak": init_class_leak,
        "train_class_leak": train_class_leak,
        "use_output_calibration": use_output_calibration,
        "init_calibration_temperature": init_calibration_temperature,
        "train_calibration": train_calibration,
        "theta_prune_threshold": theta_prune_threshold,
        "theta_l1_weight": theta_l1_weight,
        "class_leak_l1_weight": class_leak_l1_weight,
        "calibration_l2_weight": calibration_l2_weight,
    }

    for dataset_name in datasets:
        X, y = load_builtin_dataset(dataset_name)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        print(f"[dataset] {dataset_name}: {len(X)} samples, {X.shape[1]} features", flush=True)
        lines.append(f"=== Dataset: {dataset_name} ===")
        flush_output()

        for fold_idx, (trainval_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            if max_folds is not None and fold_idx > max_folds:
                break

            X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval,
                y_trainval,
                test_size=0.2,
                random_state=seed + fold_idx,
                stratify=y_trainval,
            )
            fold_seed = seed + fold_idx

            fold_models: list[tuple[str, callable]] = []
            if include_baselines:
                fold_models.extend(
                    [
                        (
                            "ExtraTrees",
                            lambda fold_seed=fold_seed: evaluate_extratrees(
                                X_train,
                                y_train,
                                X_test,
                                seed=fold_seed,
                            ),
                        ),
                        (
                            "BranchNet-Neural (frozen W2)",
                            lambda fold_seed=fold_seed: evaluate_branchnet_neural(
                                X_train,
                                y_train,
                                X_val,
                                y_val,
                                X_test,
                                seed=fold_seed,
                                epochs=branchnet_epochs,
                                learning_rate=branchnet_lr,
                                device=device,
                            ),
                        ),
                    ]
                )

            for spec in specs:
                fold_models.append(
                    (
                        spec.name,
                        lambda spec=spec, fold_seed=fold_seed: evaluate_rita_e2e(
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            seed=fold_seed,
                            branchnet_epochs=branchnet_epochs,
                            branchnet_lr=branchnet_lr,
                            e2e_epochs=e2e_epochs,
                            e2e_lr=e2e_lr,
                            loss_mode=spec.loss_mode,
                            train_w1=spec.train_w1,
                            head_type=spec.head_type,
                            theta_init_mode=spec.theta_init_mode,
                            learnable_competition=spec.learnable_competition,
                            use_posterior=spec.use_posterior,
                            train_reliability=train_reliability,
                            p_high=p_high,
                            p_low=p_low,
                            branch_truth_loss_weight=branch_truth_loss_weight,
                            **head_kwargs,
                            device=device,
                        ),
                    )
                )

            lines.append(f"-- Fold {fold_idx} --")
            print(
                f"[fold] dataset={dataset_name} fold={fold_idx}/{folds_to_run} "
                f"train={len(X_train)} val={len(X_val)} test={len(X_test)}",
                flush=True,
            )
            flush_output()

            for model_name, runner in fold_models:
                task_idx = completed_tasks + 1
                elapsed_before = time.perf_counter() - benchmark_start
                eta_before = (
                    np.mean(task_durations) * (total_tasks - completed_tasks)
                    if task_durations
                    else 0.0
                )
                print(
                    f"[task {task_idx}/{total_tasks}] "
                    f"dataset={dataset_name} fold={fold_idx} model={model_name} "
                    f"(elapsed={format_duration(elapsed_before)}, eta~{format_duration(eta_before)})",
                    flush=True,
                )
                task_start = time.perf_counter()
                probs = runner()
                task_duration = time.perf_counter() - task_start
                task_durations.append(task_duration)
                completed_tasks += 1
                metrics = compute_metrics(y_test, probs)
                records.append(
                    {
                        "dataset": dataset_name,
                        "fold": fold_idx,
                        "section": "baseline" if "Rita" not in model_name else "ablation",
                        "model": model_name,
                        **metrics,
                    }
                )
                lines.append(f"{model_name}: {format_metrics(metrics)}")
                flush_output()
                elapsed_after = time.perf_counter() - benchmark_start
                mean_task_time = float(np.mean(task_durations))
                remaining_tasks = total_tasks - completed_tasks
                eta_after = mean_task_time * remaining_tasks
                print(
                    f"[done {completed_tasks}/{total_tasks}] {model_name} "
                    f"in {format_duration(task_duration)} -> {format_metrics(metrics)} "
                    f"(elapsed={format_duration(elapsed_after)}, eta~{format_duration(eta_after)})",
                    flush=True,
                )

        lines.append("")
        flush_output()

    summary = summarize_records(records)
    lines.append("=== Summary ===")
    for row in summary:
        lines.append(f"{row['model']}: {format_metrics(row)}")
    flush_output()

    total_elapsed = time.perf_counter() - benchmark_start
    print(
        f"[ablation-matrix] finished in {format_duration(total_elapsed)}; "
        f"results saved to {output_file}",
        flush=True,
    )
    return records, summary, output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--use-learnable-competition", action="store_true")
    parser.add_argument("--freeze-reliability", action="store_true")
    parser.add_argument("--p-high", type=float, default=0.95)
    parser.add_argument("--p-low", type=float, default=0.05)
    parser.add_argument("--branch-truth-loss-weight", type=float, default=0.0)
    parser.add_argument("--use-class-leak", action="store_true")
    parser.add_argument("--init-class-leak", type=float, default=0.0)
    parser.add_argument("--freeze-class-leak", action="store_true")
    parser.add_argument("--use-output-calibration", action="store_true")
    parser.add_argument("--init-calibration-temperature", type=float, default=1.0)
    parser.add_argument("--freeze-calibration", action="store_true")
    parser.add_argument("--theta-prune-threshold", type=float, default=0.0)
    parser.add_argument("--theta-l1-weight", type=float, default=0.0)
    parser.add_argument("--class-leak-l1-weight", type=float, default=0.0)
    parser.add_argument("--calibration-l2-weight", type=float, default=0.0)
    parser.add_argument("--branchnet-epochs", type=int, default=20)
    parser.add_argument("--branchnet-lr", type=float, default=1e-2)
    parser.add_argument("--e2e-epochs", type=int, default=20)
    parser.add_argument("--e2e-lr", type=float, default=1e-3)
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(BENCHMARKS_DIR / "ablation_matrix_results.txt"),
    )
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = build_default_ablation_specs(
        use_learnable_competition=args.use_learnable_competition,
    )
    _, _, output_path = run_ablation_matrix(
        datasets=args.datasets,
        experiment_specs=specs,
        folds=args.folds,
        seed=args.seed,
        include_baselines=not args.no_baselines,
        train_reliability=not args.freeze_reliability,
        p_high=args.p_high,
        p_low=args.p_low,
        branch_truth_loss_weight=args.branch_truth_loss_weight,
        use_class_leak=args.use_class_leak,
        init_class_leak=args.init_class_leak,
        train_class_leak=not args.freeze_class_leak,
        use_output_calibration=args.use_output_calibration,
        init_calibration_temperature=args.init_calibration_temperature,
        train_calibration=not args.freeze_calibration,
        theta_prune_threshold=args.theta_prune_threshold,
        theta_l1_weight=args.theta_l1_weight,
        class_leak_l1_weight=args.class_leak_l1_weight,
        calibration_l2_weight=args.calibration_l2_weight,
        branchnet_epochs=args.branchnet_epochs,
        branchnet_lr=args.branchnet_lr,
        e2e_epochs=args.e2e_epochs,
        e2e_lr=args.e2e_lr,
        device=args.device,
        output_path=args.output_path,
        max_folds=args.max_folds,
    )
    print(f"Saved ablation matrix results to {output_path}")


if __name__ == "__main__":
    main()
