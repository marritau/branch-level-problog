import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import BranchNetFramwork
from sweep_posterior_reliability import sweep_posterior_reliability


def test_sweep_posterior_reliability_smoke():
    with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        output_path = tmp_dir / "posterior_reliability_sweep.txt"
        run_outputs_dir = tmp_dir / "runs"
        temporal_checkpoint = tmp_dir / "temporal_posterior_sweep.pt"
        original_checkpoint = BranchNetFramwork.TEMPORAL_CHECKPOINT_PATH
        BranchNetFramwork.TEMPORAL_CHECKPOINT_PATH = temporal_checkpoint
        try:
            results, saved_path = sweep_posterior_reliability(
                datasets=("iris",),
                folds=2,
                seed=42,
                p_high_values=(0.90, 0.95),
                p_low_values=(0.05,),
                branchnet_epochs=2,
                e2e_epochs=2,
                output_path=output_path,
                run_outputs_dir=run_outputs_dir,
                max_folds=1,
            )
        finally:
            BranchNetFramwork.TEMPORAL_CHECKPOINT_PATH = original_checkpoint

        assert saved_path == output_path
        assert output_path.exists()
        assert len(results) == 4
        assert all("Posterior" in row["model"] for row in results)
        assert all("p_high" in row and "p_low" in row for row in results)
        run_files = list(run_outputs_dir.glob("*.txt"))
        assert len(run_files) == 2

        text = output_path.read_text(encoding="utf-8")
        assert "# Posterior reliability sweep" in text
        assert "=== p_high=0.9000, p_low=0.0500 ===" in text
        assert "=== Best by log_loss ===" in text

    print("test_sweep_posterior_reliability_smoke OK")


if __name__ == "__main__":
    test_sweep_posterior_reliability_smoke()
