import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import BranchNetFramwork
from compare_e2e import compare_e2e_benchmark


def test_compare_e2e_smoke():
    with tempfile.TemporaryDirectory(dir=ROOT_DIR) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        output_path = tmp_dir / "compare_e2e_results_smoke.txt"
        temporal_checkpoint = tmp_dir / "temporal_smoke.pt"
        original_checkpoint = BranchNetFramwork.TEMPORAL_CHECKPOINT_PATH
        BranchNetFramwork.TEMPORAL_CHECKPOINT_PATH = temporal_checkpoint
        try:
            summary, ablation_summary, saved_path = compare_e2e_benchmark(
                datasets=("iris",),
                folds=2,
                seed=42,
                include_w1_ablation=False,
                save_reliability_diagrams=True,
                reliability_dir=tmp_dir / "reliability",
                branchnet_epochs=2,
                e2e_epochs=2,
                output_path=output_path,
                max_folds=1,
            )
        finally:
            BranchNetFramwork.TEMPORAL_CHECKPOINT_PATH = original_checkpoint

        assert saved_path == output_path
        assert output_path.exists()
        assert any(row["model"] == "ExtraTrees" for row in summary)
        assert any(row["model"] == "BranchNet-Neural (frozen W2)" for row in summary)
        assert any(row["model"] == "Rita-e2e-NoisyOr-BCE" for row in summary)
        assert any(row["model"] == "Rita-e2e-NoisyOr-NLL" for row in summary)
        assert all("brier_score" in row for row in summary)
        assert all("ece" in row for row in summary)
        assert ablation_summary == []

        text = output_path.read_text(encoding="utf-8")
        assert "=== Dataset: iris ===" in text
        assert "=== Main Summary ===" in text
        assert "brier_score=" in text
        assert "ece=" in text
        assert "=== Reliability Diagrams ===" in text
        diagram_paths = list((tmp_dir / "reliability").glob("*.png"))
        assert len(diagram_paths) == 4

    print("test_compare_e2e_smoke OK")


if __name__ == "__main__":
    test_compare_e2e_smoke()
