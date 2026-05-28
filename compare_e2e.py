import argparse
from pathlib import Path
import re
import time
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split

from BranchNetFramwork import BranchNetModel
from calibration_metrics import (
    expected_calibration_error,
    multiclass_brier_score,
    save_top_label_reliability_diagram,
)
from project_paths import BENCHMARKS_DIR, RELIABILITY_DIR
from train_e2e import predict_e2e_class_probs, train_e2e


DEFAULT_DATASETS = ("iris", "wine", "breast_cancer", "digits")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def compute_tree_config(n_features: int, n_labels: int) -> dict:
    depth_proxy = round(np.log2(n_features)) + 4
    n_estimators = n_labels + round(np.log2(n_features))
    max_leaf_nodes = 2 ** depth_proxy
    return {
        "n_estimators": n_estimators,
        "max_leaf_nodes": max_leaf_nodes,
    }


def load_builtin_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    loaders: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {
        "iris": lambda: load_iris(return_X_y=True),
        "wine": lambda: load_wine(return_X_y=True),
        "breast_cancer": lambda: load_breast_cancer(return_X_y=True),
        "digits": lambda: load_digits(return_X_y=True),
    }
    if name not in loaders:
        raise ValueError(f"Unsupported dataset: {name}")

    X, y = loaders[name]()
    return X.astype(np.float32), y.astype(np.int64)


def normalize_probs_numpy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, 1.0) + eps
    probs_sum = probs.sum(axis=1, keepdims=True)
    return probs / probs_sum


def make_safe_filename(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, calibration_bins: int = 10) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = normalize_probs_numpy(np.asarray(probs, dtype=np.float64))
    preds = np.argmax(probs, axis=1)
    labels = list(range(probs.shape[1]))
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted")),
        "mcc": float(matthews_corrcoef(y_true, preds)),
        "log_loss": float(log_loss(y_true, probs, labels=labels)),
        "brier_score": float(multiclass_brier_score(y_true, probs)),
        "ece": float(expected_calibration_error(y_true, probs, n_bins=calibration_bins)),
    }


def build_branchnet_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    epochs: int,
    learning_rate: float,
    device: str = "cpu",
) -> BranchNetModel:
    n_samples, n_features = X_train.shape
    n_labels = len(np.unique(y_train))
    config = compute_tree_config(n_features=n_features, n_labels=n_labels)
    forest = ExtraTreesClassifier(random_state=seed, **config)
    forest.fit(X_train, y_train)

    model = BranchNetModel(device=device)
    model.build_model_from_ensemble(forest)
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        learning_rate=learning_rate,
        epochs=epochs,
    )
    return model


def evaluate_extratrees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    config = compute_tree_config(n_features=X_train.shape[1], n_labels=len(np.unique(y_train)))
    forest = ExtraTreesClassifier(random_state=seed, **config)
    forest.fit(X_train, y_train)
    return forest.predict_proba(X_test)


def evaluate_branchnet_neural(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    epochs: int,
    learning_rate: float,
    device: str = "cpu",
) -> np.ndarray:
    model = build_branchnet_model(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    return model.predict_proba(X_test).detach().cpu().numpy()


def evaluate_rita_e2e(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    branchnet_epochs: int,
    branchnet_lr: float,
    e2e_epochs: int,
    e2e_lr: float,
    loss_mode: str,
    train_w1: bool,
    head_type: str = "noisy_or",
    theta_init_mode: str = "weighted",
    learnable_competition: bool = False,
    use_posterior: bool = False,
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
    device: str = "cpu",
) -> np.ndarray:
    model = build_branchnet_model(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=seed,
        epochs=branchnet_epochs,
        learning_rate=branchnet_lr,
        device=device,
    )
    result = train_e2e(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        head_type=head_type,
        theta_init_mode=theta_init_mode,
        learnable_competition=learnable_competition,
        loss_mode=loss_mode,
        train_w1=train_w1,
        use_posterior=use_posterior,
        train_reliability=train_reliability,
        p_high=p_high,
        p_low=p_low,
        branch_truth_loss_weight=branch_truth_loss_weight,
        use_class_leak=use_class_leak,
        init_class_leak=init_class_leak,
        train_class_leak=train_class_leak,
        use_output_calibration=use_output_calibration,
        init_calibration_temperature=init_calibration_temperature,
        train_calibration=train_calibration,
        theta_prune_threshold=theta_prune_threshold,
        theta_l1_weight=theta_l1_weight,
        class_leak_l1_weight=class_leak_l1_weight,
        calibration_l2_weight=calibration_l2_weight,
        learning_rate=e2e_lr,
        epochs=e2e_epochs,
        patience=min(50, e2e_epochs),
        batch_size=256,
    )
    probs = predict_e2e_class_probs(
        result.model,
        result.head,
        X_test,
        posterior_layer=result.posterior_layer,
        normalize_for_nll=(loss_mode == "nll"),
    ).detach().cpu().numpy()
    if loss_mode == "bce":
        probs = normalize_probs_numpy(probs)
    return probs


def format_metrics(metrics: dict) -> str:
    return (
        f"accuracy={metrics['accuracy']:.6f}, "
        f"weighted_f1={metrics['weighted_f1']:.6f}, "
        f"mcc={metrics['mcc']:.6f}, "
        f"log_loss={metrics['log_loss']:.6f}, "
        f"brier_score={metrics['brier_score']:.6f}, "
        f"ece={metrics['ece']:.6f}"
    )


def summarize_records(records: list[dict]) -> list[dict]:
    summary = []
    keys = ("accuracy", "weighted_f1", "mcc", "log_loss", "brier_score", "ece")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in records:
        grouped.setdefault((row["section"], row["model"]), []).append(row)

    for (section, model), rows in grouped.items():
        aggregated = {
            metric: float(np.mean([r[metric] for r in rows]))
            for metric in keys
        }
        aggregated.update(
            {
                "section": section,
                "model": model,
                "folds": len(rows),
            }
        )
        summary.append(aggregated)
    return sorted(summary, key=lambda r: (r["section"], r["model"]))


def compare_e2e_benchmark(
    datasets: Iterable[str] = DEFAULT_DATASETS,
    folds: int = 5,
    seed: int = 42,
    include_w1_ablation: bool = True,
    include_posterior: bool = False,
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
    head_type: str = "noisy_or",
    theta_init_mode: str = "weighted",
    learnable_competition: bool = False,
    reliability_bins: int = 10,
    save_reliability_diagrams: bool = False,
    reliability_dir: Optional[str | Path] = None,
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

    output_file = Path(output_path) if output_path is not None else BENCHMARKS_DIR / "compare_e2e_results.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    reliability_output_dir = None
    if save_reliability_diagrams:
        reliability_output_dir = (
            Path(reliability_dir)
            if reliability_dir is not None
            else RELIABILITY_DIR / output_file.stem
        )
        reliability_output_dir.mkdir(parents=True, exist_ok=True)
    if head_type == "noisy_or":
        head_label = "NoisyOr"
    elif head_type == "softmax_competition":
        head_label = "SoftmaxCompetition"
    else:
        raise ValueError(
            f"Unsupported head_type: {head_type}. Expected 'noisy_or' or 'softmax_competition'."
        )
    if theta_init_mode not in {"weighted", "normalized"}:
        raise ValueError(
            f"Unsupported theta_init_mode: {theta_init_mode}. "
            "Expected 'weighted' or 'normalized'."
        )
    theta_label = "" if theta_init_mode == "weighted" else "-NormTheta"

    records: list[dict] = []
    ablations: list[dict] = []
    lines: list[str] = []
    reliability_payloads: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = {}
    datasets = list(datasets)
    folds_to_run = min(folds, max_folds) if max_folds is not None else folds
    models_per_fold = 4
    if include_posterior:
        models_per_fold += 2
    if include_w1_ablation:
        models_per_fold += 2
        if include_posterior:
            models_per_fold += 2
    total_tasks = len(datasets) * folds_to_run * models_per_fold
    completed_tasks = 0
    task_durations: list[float] = []
    benchmark_start = time.perf_counter()

    def flush_output() -> None:
        output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"[benchmark] starting: datasets={datasets}, folds={folds_to_run}, "
        f"models_per_fold={models_per_fold}, total_tasks={total_tasks}",
        flush=True,
    )
    lines.append(
        f"# Benchmark start: datasets={datasets}, folds={folds_to_run}, "
        f"models_per_fold={models_per_fold}, total_tasks={total_tasks}"
    )
    flush_output()

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
            fold_models = [
                ("main", "ExtraTrees", lambda: evaluate_extratrees(X_train, y_train, X_test, seed=fold_seed)),
                (
                    "main",
                    "BranchNet-Neural (frozen W2)",
                    lambda: evaluate_branchnet_neural(
                        X_train, y_train, X_val, y_val, X_test,
                        seed=fold_seed,
                        epochs=branchnet_epochs,
                        learning_rate=branchnet_lr,
                        device=device,
                    ),
                ),
                (
                    "main",
                    f"Rita-e2e-{head_label}{theta_label}-BCE",
                    lambda: evaluate_rita_e2e(
                        X_train, y_train, X_val, y_val, X_test,
                        seed=fold_seed,
                        branchnet_epochs=branchnet_epochs,
                        branchnet_lr=branchnet_lr,
                        e2e_epochs=e2e_epochs,
                        e2e_lr=e2e_lr,
                        head_type=head_type,
                        theta_init_mode=theta_init_mode,
                        learnable_competition=learnable_competition,
                        loss_mode="bce",
                        train_w1=False,
                        branch_truth_loss_weight=branch_truth_loss_weight,
                        **head_kwargs,
                        device=device,
                    ),
                ),
                (
                    "main",
                    f"Rita-e2e-{head_label}{theta_label}-NLL",
                    lambda: evaluate_rita_e2e(
                        X_train, y_train, X_val, y_val, X_test,
                        seed=fold_seed,
                        branchnet_epochs=branchnet_epochs,
                        branchnet_lr=branchnet_lr,
                        e2e_epochs=e2e_epochs,
                        e2e_lr=e2e_lr,
                        head_type=head_type,
                        theta_init_mode=theta_init_mode,
                        learnable_competition=learnable_competition,
                        loss_mode="nll",
                        train_w1=False,
                        branch_truth_loss_weight=branch_truth_loss_weight,
                        **head_kwargs,
                        device=device,
                    ),
                ),
            ]

            if include_posterior:
                fold_models.extend(
                    [
                        (
                            "main",
                            f"Rita-e2e-Posterior-{head_label}{theta_label}-BCE",
                            lambda: evaluate_rita_e2e(
                                X_train, y_train, X_val, y_val, X_test,
                                seed=fold_seed,
                                branchnet_epochs=branchnet_epochs,
                                branchnet_lr=branchnet_lr,
                                e2e_epochs=e2e_epochs,
                                e2e_lr=e2e_lr,
                                head_type=head_type,
                                theta_init_mode=theta_init_mode,
                                learnable_competition=learnable_competition,
                                loss_mode="bce",
                                train_w1=False,
                                use_posterior=True,
                                train_reliability=train_reliability,
                                p_high=p_high,
                                p_low=p_low,
                                branch_truth_loss_weight=branch_truth_loss_weight,
                                **head_kwargs,
                                device=device,
                            ),
                        ),
                        (
                            "main",
                            f"Rita-e2e-Posterior-{head_label}{theta_label}-NLL",
                            lambda: evaluate_rita_e2e(
                                X_train, y_train, X_val, y_val, X_test,
                                seed=fold_seed,
                                branchnet_epochs=branchnet_epochs,
                                branchnet_lr=branchnet_lr,
                                e2e_epochs=e2e_epochs,
                                e2e_lr=e2e_lr,
                                head_type=head_type,
                                theta_init_mode=theta_init_mode,
                                learnable_competition=learnable_competition,
                                loss_mode="nll",
                                train_w1=False,
                                use_posterior=True,
                                train_reliability=train_reliability,
                                p_high=p_high,
                                p_low=p_low,
                                branch_truth_loss_weight=branch_truth_loss_weight,
                                **head_kwargs,
                                device=device,
                            ),
                        ),
                    ]
                )

            if include_w1_ablation:
                ablation_models = [
                    (
                        "ablation",
                        f"Rita-e2e-{head_label}{theta_label}-BCE (train_w1=True)",
                        lambda: evaluate_rita_e2e(
                            X_train, y_train, X_val, y_val, X_test,
                            seed=fold_seed,
                            branchnet_epochs=branchnet_epochs,
                            branchnet_lr=branchnet_lr,
                            e2e_epochs=e2e_epochs,
                            e2e_lr=e2e_lr,
                            head_type=head_type,
                            theta_init_mode=theta_init_mode,
                            learnable_competition=learnable_competition,
                            loss_mode="bce",
                            train_w1=True,
                            branch_truth_loss_weight=branch_truth_loss_weight,
                            **head_kwargs,
                            device=device,
                        ),
                    ),
                    (
                        "ablation",
                        f"Rita-e2e-{head_label}{theta_label}-NLL (train_w1=True)",
                        lambda: evaluate_rita_e2e(
                            X_train, y_train, X_val, y_val, X_test,
                            seed=fold_seed,
                            branchnet_epochs=branchnet_epochs,
                            branchnet_lr=branchnet_lr,
                            e2e_epochs=e2e_epochs,
                            e2e_lr=e2e_lr,
                            head_type=head_type,
                            theta_init_mode=theta_init_mode,
                            learnable_competition=learnable_competition,
                            loss_mode="nll",
                            train_w1=True,
                            branch_truth_loss_weight=branch_truth_loss_weight,
                            **head_kwargs,
                            device=device,
                        ),
                    ),
                ]
                if include_posterior:
                    ablation_models.extend(
                        [
                            (
                                "ablation",
                                f"Rita-e2e-Posterior-{head_label}{theta_label}-BCE (train_w1=True)",
                                lambda: evaluate_rita_e2e(
                                    X_train, y_train, X_val, y_val, X_test,
                                    seed=fold_seed,
                                    branchnet_epochs=branchnet_epochs,
                                    branchnet_lr=branchnet_lr,
                                    e2e_epochs=e2e_epochs,
                                    e2e_lr=e2e_lr,
                                    head_type=head_type,
                                    theta_init_mode=theta_init_mode,
                                    learnable_competition=learnable_competition,
                                    loss_mode="bce",
                                    train_w1=True,
                                    use_posterior=True,
                                    train_reliability=train_reliability,
                                    p_high=p_high,
                                    p_low=p_low,
                                    branch_truth_loss_weight=branch_truth_loss_weight,
                                    **head_kwargs,
                                    device=device,
                                ),
                            ),
                            (
                                "ablation",
                                f"Rita-e2e-Posterior-{head_label}{theta_label}-NLL (train_w1=True)",
                                lambda: evaluate_rita_e2e(
                                    X_train, y_train, X_val, y_val, X_test,
                                    seed=fold_seed,
                                    branchnet_epochs=branchnet_epochs,
                                    branchnet_lr=branchnet_lr,
                                    e2e_epochs=e2e_epochs,
                                    e2e_lr=e2e_lr,
                                    head_type=head_type,
                                    theta_init_mode=theta_init_mode,
                                    learnable_competition=learnable_competition,
                                    loss_mode="nll",
                                    train_w1=True,
                                    use_posterior=True,
                                    train_reliability=train_reliability,
                                    p_high=p_high,
                                    p_low=p_low,
                                    branch_truth_loss_weight=branch_truth_loss_weight,
                                    **head_kwargs,
                                    device=device,
                                ),
                            ),
                        ]
                    )
                fold_models.extend(ablation_models)

            lines.append(f"-- Fold {fold_idx} --")
            print(
                f"[fold] dataset={dataset_name} fold={fold_idx}/{folds_to_run} "
                f"train={len(X_train)} val={len(X_val)} test={len(X_test)}",
                flush=True,
            )
            flush_output()
            for section, model_name, runner in fold_models:
                task_idx = completed_tasks + 1
                elapsed_before = time.perf_counter() - benchmark_start
                eta_before = (
                    np.mean(task_durations) * (total_tasks - completed_tasks)
                    if task_durations
                    else 0.0
                )
                print(
                    f"[task {task_idx}/{total_tasks}] "
                    f"dataset={dataset_name} fold={fold_idx} section={section} model={model_name} "
                    f"(elapsed={format_duration(elapsed_before)}, eta~{format_duration(eta_before)})",
                    flush=True,
                )
                task_start = time.perf_counter()
                probs = runner()
                normalized_probs = normalize_probs_numpy(probs)
                task_duration = time.perf_counter() - task_start
                task_durations.append(task_duration)
                completed_tasks += 1
                metrics = compute_metrics(y_test, normalized_probs, calibration_bins=reliability_bins)
                record = {
                    "dataset": dataset_name,
                    "fold": fold_idx,
                    "section": section,
                    "model": model_name,
                    **metrics,
                }
                if section == "main":
                    records.append(record)
                else:
                    ablations.append(record)
                if save_reliability_diagrams:
                    payload = reliability_payloads.setdefault(
                        (dataset_name, section, model_name),
                        {"y_true": [], "probs": []},
                    )
                    payload["y_true"].append(np.asarray(y_test, dtype=np.int64).copy())
                    payload["probs"].append(np.asarray(normalized_probs, dtype=np.float64).copy())
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
    ablation_summary = summarize_records(ablations)

    lines.append("=== Main Summary ===")
    for row in summary:
        lines.append(f"{row['model']}: {format_metrics(row)}")

    if ablation_summary:
        lines.append("")
        lines.append("=== Ablation Summary ===")
        for row in ablation_summary:
            lines.append(f"{row['model']}: {format_metrics(row)}")

    if save_reliability_diagrams and reliability_output_dir is not None:
        lines.append("")
        lines.append("=== Reliability Diagrams ===")
        for (dataset_name, section, model_name), payload in sorted(reliability_payloads.items()):
            y_true = np.concatenate(payload["y_true"], axis=0)
            probs = np.concatenate(payload["probs"], axis=0)
            diagram_name = (
                f"{make_safe_filename(dataset_name)}__"
                f"{make_safe_filename(section)}__"
                f"{make_safe_filename(model_name)}.png"
            )
            diagram_path = reliability_output_dir / diagram_name
            save_top_label_reliability_diagram(
                y_true,
                probs,
                diagram_path,
                title=f"{dataset_name} | {model_name}",
                n_bins=reliability_bins,
            )
            lines.append(
                f"{dataset_name} | {section} | {model_name}: {diagram_path}"
            )

    flush_output()
    total_elapsed = time.perf_counter() - benchmark_start
    print(
        f"[benchmark] finished in {format_duration(total_elapsed)}; results saved to {output_file}",
        flush=True,
    )
    return summary, ablation_summary, output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--include-w1-ablation", action="store_true")
    parser.add_argument("--include-posterior", action="store_true")
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
    parser.add_argument("--head-type", type=str, default="noisy_or")
    parser.add_argument("--theta-init-mode", type=str, default="weighted")
    parser.add_argument("--learnable-competition", action="store_true")
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument("--save-reliability-diagrams", action="store_true")
    parser.add_argument("--reliability-dir", type=str, default=None)
    parser.add_argument("--branchnet-epochs", type=int, default=20)
    parser.add_argument("--branchnet-lr", type=float, default=1e-2)
    parser.add_argument("--e2e-epochs", type=int, default=20)
    parser.add_argument("--e2e-lr", type=float, default=1e-3)
    parser.add_argument("--output-path", type=str, default=str(BENCHMARKS_DIR / "compare_e2e_results.txt"))
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, output_path = compare_e2e_benchmark(
        datasets=args.datasets,
        folds=args.folds,
        seed=args.seed,
        include_w1_ablation=args.include_w1_ablation,
        include_posterior=args.include_posterior,
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
        head_type=args.head_type,
        theta_init_mode=args.theta_init_mode,
        learnable_competition=args.learnable_competition,
        reliability_bins=args.reliability_bins,
        save_reliability_diagrams=args.save_reliability_diagrams,
        reliability_dir=args.reliability_dir,
        branchnet_epochs=args.branchnet_epochs,
        branchnet_lr=args.branchnet_lr,
        e2e_epochs=args.e2e_epochs,
        e2e_lr=args.e2e_lr,
        device=args.device,
        output_path=args.output_path,
        max_folds=args.max_folds,
    )
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
