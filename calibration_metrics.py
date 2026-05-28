from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs must have shape [n_samples, n_classes], got {probs.shape}")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_true has {y_true.shape[0]} rows but probs has {probs.shape[0]} rows"
        )
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def top_label_reliability_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs must have shape [n_samples, n_classes], got {probs.shape}")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_true has {y_true.shape[0]} rows but probs has {probs.shape[0]} rows"
        )
    if n_bins <= 1:
        raise ValueError(f"n_bins must be greater than 1, got {n_bins}")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1], right=True)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.zeros(n_bins, dtype=np.int64)
    mean_confidence = np.zeros(n_bins, dtype=np.float64)
    accuracy = np.zeros(n_bins, dtype=np.float64)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        counts[bin_idx] = int(mask.sum())
        if counts[bin_idx] > 0:
            mean_confidence[bin_idx] = float(confidences[mask].mean())
            accuracy[bin_idx] = float(correctness[mask].mean())

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "counts": counts,
        "mean_confidence": mean_confidence,
        "accuracy": accuracy,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    curve = top_label_reliability_curve(y_true, probs, n_bins=n_bins)
    counts = curve["counts"].astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    weights = counts / total
    gaps = np.abs(curve["accuracy"] - curve["mean_confidence"])
    return float(np.sum(weights * gaps))


def save_top_label_reliability_diagram(
    y_true: np.ndarray,
    probs: np.ndarray,
    output_path: str | Path,
    title: Optional[str] = None,
    n_bins: int = 10,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    curve = top_label_reliability_curve(y_true, probs, n_bins=n_bins)
    bin_edges = curve["bin_edges"]
    bin_centers = curve["bin_centers"]
    accuracy = curve["accuracy"]
    mean_confidence = curve["mean_confidence"]
    counts = curve["counts"].astype(np.float64)
    widths = np.diff(bin_edges)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        bin_centers,
        accuracy,
        width=widths * 0.9,
        color="#4C78A8",
        alpha=0.75,
        edgecolor="white",
        label="Accuracy",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#E45756", linewidth=1.5, label="Perfect calibration")
    ax.plot(bin_centers, mean_confidence, color="#72B7B2", marker="o", linewidth=1.5, label="Mean confidence")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    if title:
        ax.set_title(title)
    ece = expected_calibration_error(y_true, probs, n_bins=n_bins)
    brier = multiclass_brier_score(y_true, probs)
    ax.text(
        0.02,
        0.98,
        f"ECE={ece:.4f}\nBrier={brier:.4f}\nN={int(counts.sum())}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#CCCCCC"),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
