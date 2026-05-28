import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import BranchNetFramwork
from ablation_matrix import ExperimentSpec, build_default_ablation_specs, run_ablation_matrix


def test_build_default_ablation_specs_contains_expected_axes():
    specs = build_default_ablation_specs()
    assert len(specs) == 16
    names = {spec.name for spec in specs}
    assert "BranchNet-e2e-NoisyOr-Weighted-BCE" in names
    assert "BranchNet-e2e-NoisyOr-NormTheta-BCE-Posterior-trainW1" in names
    assert "BranchNet-e2e-SoftmaxCompetition-Weighted-NLL" in names
    assert "BranchNet-e2e-SoftmaxCompetition-NormTheta-NLL-Posterior-trainW1" in names


def test_run_ablation_matrix_smoke():
    specs = [
        ExperimentSpec(
            name="BranchNet-e2e-NoisyOr-Weighted-BCE",
            head_type="noisy_or",
            theta_init_mode="weighted",
            loss_mode="bce",
            use_posterior=False,
            train_w1=False,
        ),
        ExperimentSpec(
            name="BranchNet-e2e-SoftmaxCompetition-NormTheta-NLL-Posterior-trainW1",
            head_type="softmax_competition",
            theta_init_mode="normalized",
            loss_mode="nll",
            use_posterior=True,
            train_w1=True,
        ),
    ]

    with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        output_path = tmp_dir / "ablation_matrix_results.txt"
        temporal_checkpoint = tmp_dir / "temporal_ablation.pt"
        original_checkpoint = BranchNetFramwork.TEMPORAL_CHECKPOINT_PATH
        BranchNetFramwork.TEMPORAL_CHECKPOINT_PATH = temporal_checkpoint
        try:
            records, summary, saved_path = run_ablation_matrix(
                datasets=("iris",),
                experiment_specs=specs,
                folds=2,
                seed=42,
                branchnet_epochs=2,
                e2e_epochs=2,
                output_path=output_path,
                max_folds=1,
            )
        finally:
            BranchNetFramwork.TEMPORAL_CHECKPOINT_PATH = original_checkpoint

        assert saved_path == output_path
        assert output_path.exists()
        assert len(records) == 4
        summary_names = {row["model"] for row in summary}
        assert "ExtraTrees" in summary_names
        assert "BranchNet-Neural (frozen W2)" in summary_names
        assert "BranchNet-e2e-NoisyOr-Weighted-BCE" in summary_names
        assert "BranchNet-e2e-SoftmaxCompetition-NormTheta-NLL-Posterior-trainW1" in summary_names

        text = output_path.read_text(encoding="utf-8")
        assert "=== Experiment Specs ===" in text
        assert "=== Summary ===" in text
        assert "brier_score=" in text
        assert "ece=" in text

    print("test_ablation_matrix_smoke OK")


if __name__ == "__main__":
    test_build_default_ablation_specs_contains_expected_axes()
    test_run_ablation_matrix_smoke()
