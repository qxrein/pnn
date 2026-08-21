"""End-to-end smoke test: short training, evaluation, figures."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.evaluate import evaluate_pinn
from src.train import train_pinn
from src.utils import resolve_device
from src.visualization import generate_all_figures


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_smoke_training_evaluation(project_root: Path, tmp_path_factory) -> None:
    config = load_config(project_root / "configs/default.yaml", smoke=True)
    out = tmp_path_factory.mktemp("smoke_outputs")
    config.paths.output_dir = str(out)
    config.paths.checkpoint_dir = str(out / "checkpoints")
    config.paths.figure_dir = str(out / "figures")
    config.paths.history_file = str(out / "training_history.csv")
    config.paths.results_file = str(out / "results.npz")
    config.paths.metrics_file = str(out / "metrics.json")
    config.paths.config_snapshot = str(out / "run_config.yaml")
    config.training.checkpoint_interval = 25
    config.training.validation_interval = 10
    config.training.epochs = 50
    config.training.early_stopping_patience = 0
    config.training.device = "cpu"  # deterministic smoke test on CPU

    summary = train_pinn(config)
    assert Path(summary["best_checkpoint"]).exists()
    assert Path(summary["history_file"]).exists()

    eval_result = evaluate_pinn(config, summary["best_checkpoint"])
    assert Path(eval_result["results_file"]).exists()
    assert Path(eval_result["metrics_file"]).exists()

    device = resolve_device(config.training.device)
    figures = generate_all_figures(
        results_npz=Path(eval_result["results_file"]),
        history_file=Path(summary["history_file"]),
        output_dir=Path(config.paths.figure_dir),
        model=None,
        config=config,
        device=device,
    )
    assert len(figures) >= 7
