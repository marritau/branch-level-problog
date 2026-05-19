import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from compare_e2e import compare_e2e_benchmark
from project_paths import BENCHMARKS_DIR


def test_compare_e2e_smoke():
    output_path = BENCHMARKS_DIR / "compare_e2e_results_smoke.txt"
    summary, ablation_summary, saved_path = compare_e2e_benchmark(
        datasets=("iris",),
        folds=2,
        seed=42,
        include_w1_ablation=False,
        branchnet_epochs=2,
        e2e_epochs=2,
        output_path=output_path,
        max_folds=1,
    )

    assert saved_path == output_path
    assert output_path.exists()
    assert any(row["model"] == "ExtraTrees" for row in summary)
    assert any(row["model"] == "BranchNet-Neural (frozen W2)" for row in summary)
    assert any(row["model"] == "Rita-e2e-NoisyOr-BCE" for row in summary)
    assert any(row["model"] == "Rita-e2e-NoisyOr-NLL" for row in summary)
    assert ablation_summary == []

    text = output_path.read_text(encoding="utf-8")
    assert "=== Dataset: iris ===" in text
    assert "=== Main Summary ===" in text

    print("test_compare_e2e_smoke OK")


if __name__ == "__main__":
    test_compare_e2e_smoke()
