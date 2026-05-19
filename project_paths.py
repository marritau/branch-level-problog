from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
TESTS_DIR = ROOT_DIR / "tests"
SCRIPTS_DIR = ROOT_DIR / "scripts"

OUTPUT_DIR = ROOT_DIR / "output"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
PROBLOG_OUTPUT_DIR = OUTPUT_DIR / "problog"
DEBUG_EXPORT_DIR = OUTPUT_DIR / "debug_export"
BENCHMARKS_DIR = OUTPUT_DIR / "benchmarks"

TEMPORAL_CHECKPOINT_PATH = CHECKPOINTS_DIR / "temporal.pt"


def ensure_repo_layout() -> None:
    for directory in (
        TESTS_DIR / "branch",
        TESTS_DIR / "problog",
        TESTS_DIR / "e2e",
        SCRIPTS_DIR / "diagnostics",
        CHECKPOINTS_DIR,
        PROBLOG_OUTPUT_DIR,
        DEBUG_EXPORT_DIR,
        BENCHMARKS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


ensure_repo_layout()
