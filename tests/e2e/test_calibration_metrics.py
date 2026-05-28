import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from calibration_metrics import (
    expected_calibration_error,
    multiclass_brier_score,
    top_label_reliability_curve,
)


def test_multiclass_brier_score_is_zero_for_perfect_predictions():
    y_true = np.array([0, 1, 2], dtype=np.int64)
    probs = np.eye(3, dtype=np.float64)
    score = multiclass_brier_score(y_true, probs)
    assert abs(score) < 1e-12


def test_expected_calibration_error_detects_overconfidence():
    y_true = np.array([0, 1, 1, 0], dtype=np.int64)
    probs = np.array(
        [
            [0.90, 0.10],
            [0.90, 0.10],
            [0.10, 0.90],
            [0.10, 0.90],
        ],
        dtype=np.float64,
    )
    ece = expected_calibration_error(y_true, probs, n_bins=5)
    assert ece > 0.0


def test_top_label_reliability_curve_counts_match_sample_count():
    y_true = np.array([0, 1, 1, 0], dtype=np.int64)
    probs = np.array(
        [
            [0.70, 0.30],
            [0.40, 0.60],
            [0.20, 0.80],
            [0.55, 0.45],
        ],
        dtype=np.float64,
    )
    curve = top_label_reliability_curve(y_true, probs, n_bins=4)
    assert int(curve["counts"].sum()) == y_true.shape[0]
    assert curve["bin_centers"].shape == (4,)


if __name__ == "__main__":
    test_multiclass_brier_score_is_zero_for_perfect_predictions()
    test_expected_calibration_error_detects_overconfidence()
    test_top_label_reliability_curve_counts_match_sample_count()
    print("test_calibration_metrics OK")
