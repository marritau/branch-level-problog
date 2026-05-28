import argparse
import time
from pathlib import Path
from typing import Iterable, Optional

from compare_e2e import (
    DEFAULT_DATASETS,
    compare_e2e_benchmark,
    format_duration,
    format_metrics,
)
from project_paths import BENCHMARKS_DIR


LOWER_IS_BETTER_METRICS = {"log_loss", "brier_score", "ece"}
VALID_RANKING_METRICS = {
    "accuracy",
    "weighted_f1",
    "mcc",
    "log_loss",
    "brier_score",
    "ece",
}


def _format_float_label(value: float) -> str:
    return f"{value:.4f}".replace(".", "_")


def _validate_reliability_grid(
    p_high_values: Iterable[float],
    p_low_values: Iterable[float],
) -> list[tuple[float, float]]:
    validated_pairs: list[tuple[float, float]] = []
    for p_low in p_low_values:
        p_low = float(p_low)
        if not 0.0 < p_low < 1.0:
            raise ValueError(f"p_low must be in (0, 1), got {p_low}")
        for p_high in p_high_values:
            p_high = float(p_high)
            if not 0.0 < p_high < 1.0:
                raise ValueError(f"p_high must be in (0, 1), got {p_high}")
            if p_high <= p_low:
                continue
            validated_pairs.append((p_high, p_low))
    if not validated_pairs:
        raise ValueError(
            "No valid (p_high, p_low) pairs remain after enforcing p_high > p_low"
        )
    return validated_pairs


def _is_better(candidate: dict, incumbent: Optional[dict], metric: str) -> bool:
    if incumbent is None:
        return True
    if metric in LOWER_IS_BETTER_METRICS:
        return candidate[metric] < incumbent[metric]
    return candidate[metric] > incumbent[metric]


def summarize_best_configs(
    results: list[dict],
    ranking_metric: str,
) -> list[dict]:
    best_by_model: dict[str, dict] = {}
    for row in results:
        model_name = row["model"]
        incumbent = best_by_model.get(model_name)
        if _is_better(row, incumbent, ranking_metric):
            best_by_model[model_name] = row
    return [best_by_model[key] for key in sorted(best_by_model)]


def sweep_posterior_reliability(
    datasets: Iterable[str] = DEFAULT_DATASETS,
    folds: int = 5,
    seed: int = 42,
    p_high_values: Iterable[float] = (0.90, 0.95, 0.98),
    p_low_values: Iterable[float] = (0.01, 0.05, 0.10),
    ranking_metric: str = "log_loss",
    head_type: str = "noisy_or",
    theta_init_mode: str = "weighted",
    learnable_competition: bool = False,
    train_reliability: bool = True,
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
    reliability_bins: int = 10,
    save_reliability_diagrams: bool = False,
    branchnet_epochs: int = 20,
    branchnet_lr: float = 1e-2,
    e2e_epochs: int = 20,
    e2e_lr: float = 1e-3,
    device: str = "cpu",
    output_path: Optional[str | Path] = None,
    run_outputs_dir: Optional[str | Path] = None,
    max_folds: Optional[int] = None,
) -> tuple[list[dict], Path]:
    if ranking_metric not in VALID_RANKING_METRICS:
        raise ValueError(
            f"Unsupported ranking_metric: {ranking_metric}. "
            f"Expected one of {sorted(VALID_RANKING_METRICS)}."
        )

    pairs = _validate_reliability_grid(p_high_values, p_low_values)
    output_file = (
        Path(output_path)
        if output_path is not None
        else BENCHMARKS_DIR / "posterior_reliability_sweep.txt"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    run_dir = (
        Path(run_outputs_dir)
        if run_outputs_dir is not None
        else BENCHMARKS_DIR / f"{output_file.stem}_runs"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    lines: list[str] = []
    start_time = time.perf_counter()

    def flush_output() -> None:
        output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"[posterior-sweep] starting: datasets={list(datasets)}, pairs={len(pairs)}, "
        f"ranking_metric={ranking_metric}",
        flush=True,
    )
    lines.append(
        f"# Posterior reliability sweep: datasets={list(datasets)}, pairs={len(pairs)}, "
        f"ranking_metric={ranking_metric}"
    )
    flush_output()

    for pair_idx, (p_high, p_low) in enumerate(pairs, start=1):
        elapsed = time.perf_counter() - start_time
        print(
            f"[posterior-sweep] pair {pair_idx}/{len(pairs)}: "
            f"p_high={p_high:.4f}, p_low={p_low:.4f} "
            f"(elapsed={format_duration(elapsed)})",
            flush=True,
        )
        run_label = f"p_high_{_format_float_label(p_high)}__p_low_{_format_float_label(p_low)}"
        run_output_path = run_dir / f"{run_label}.txt"
        reliability_dir = run_dir / run_label / "reliability"

        summary, _, saved_path = compare_e2e_benchmark(
            datasets=datasets,
            folds=folds,
            seed=seed,
            include_w1_ablation=False,
            include_posterior=True,
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
            head_type=head_type,
            theta_init_mode=theta_init_mode,
            learnable_competition=learnable_competition,
            reliability_bins=reliability_bins,
            save_reliability_diagrams=save_reliability_diagrams,
            reliability_dir=reliability_dir,
            branchnet_epochs=branchnet_epochs,
            branchnet_lr=branchnet_lr,
            e2e_epochs=e2e_epochs,
            e2e_lr=e2e_lr,
            device=device,
            output_path=run_output_path,
            max_folds=max_folds,
        )

        posterior_rows = [row for row in summary if "Posterior" in row["model"]]
        if not posterior_rows:
            raise RuntimeError(
                f"Posterior sweep expected posterior summary rows for p_high={p_high}, p_low={p_low}"
            )

        lines.append("")
        lines.append(f"=== p_high={p_high:.4f}, p_low={p_low:.4f} ===")
        lines.append(f"run_output: {saved_path}")
        for row in posterior_rows:
            result_row = {
                "p_high": float(p_high),
                "p_low": float(p_low),
                "run_output_path": str(saved_path),
                **row,
            }
            results.append(result_row)
            lines.append(f"{row['model']}: {format_metrics(row)}")
        flush_output()

    best_configs = summarize_best_configs(results, ranking_metric=ranking_metric)
    lines.append("")
    lines.append(f"=== Best by {ranking_metric} ===")
    for row in best_configs:
        lines.append(
            f"{row['model']}: p_high={row['p_high']:.4f}, p_low={row['p_low']:.4f}, "
            f"{format_metrics(row)}"
        )
    flush_output()

    total_elapsed = time.perf_counter() - start_time
    print(
        f"[posterior-sweep] finished in {format_duration(total_elapsed)}; "
        f"results saved to {output_file}",
        flush=True,
    )
    return results, output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--p-high-values", nargs="+", type=float, default=[0.90, 0.95, 0.98])
    parser.add_argument("--p-low-values", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument("--ranking-metric", type=str, default="log_loss")
    parser.add_argument("--freeze-reliability", action="store_true")
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
    parser.add_argument("--branchnet-epochs", type=int, default=20)
    parser.add_argument("--branchnet-lr", type=float, default=1e-2)
    parser.add_argument("--e2e-epochs", type=int, default=20)
    parser.add_argument("--e2e-lr", type=float, default=1e-3)
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(BENCHMARKS_DIR / "posterior_reliability_sweep.txt"),
    )
    parser.add_argument("--run-outputs-dir", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, output_path = sweep_posterior_reliability(
        datasets=args.datasets,
        folds=args.folds,
        seed=args.seed,
        p_high_values=args.p_high_values,
        p_low_values=args.p_low_values,
        ranking_metric=args.ranking_metric,
        head_type=args.head_type,
        theta_init_mode=args.theta_init_mode,
        learnable_competition=args.learnable_competition,
        train_reliability=not args.freeze_reliability,
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
        reliability_bins=args.reliability_bins,
        save_reliability_diagrams=args.save_reliability_diagrams,
        branchnet_epochs=args.branchnet_epochs,
        branchnet_lr=args.branchnet_lr,
        e2e_epochs=args.e2e_epochs,
        e2e_lr=args.e2e_lr,
        device=args.device,
        output_path=args.output_path,
        run_outputs_dir=args.run_outputs_dir,
        max_folds=args.max_folds,
    )
    print(f"Saved posterior reliability sweep to {output_path}")


if __name__ == "__main__":
    main()
