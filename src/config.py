"""Configuration loading and dataclass definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PhysicsConfig:
    """Physical and geometry parameters for the grating problem."""

    wavelength: float = 1.0
    n_air: float = 1.0
    n_ridge: float = 1.5
    n_substrate: float = 1.45
    period: float = 1.0
    ridge_width: float = 0.4
    ridge_height: float = 0.2
    domain_height: float = 2.0
    ridge_base_fraction: float = 0.6
    interface_margin: float = 0.02
    nx_visualization: int = 128
    nz_visualization: int = 256

    @property
    def k0(self) -> float:
        """Free-space wavenumber k0 = 2*pi / wavelength."""
        import math

        return 2.0 * math.pi / self.wavelength

    @property
    def eps_air(self) -> float:
        return self.n_air**2

    @property
    def eps_ridge(self) -> float:
        return self.n_ridge**2

    @property
    def eps_substrate(self) -> float:
        return self.n_substrate**2

    @property
    def ridge_base_z(self) -> float:
        return self.ridge_base_fraction * self.domain_height

    @property
    def ridge_x_min(self) -> float:
        return (self.period - self.ridge_width) / 2.0

    @property
    def ridge_x_max(self) -> float:
        return (self.period + self.ridge_width) / 2.0

    @property
    def ridge_z_min(self) -> float:
        return self.ridge_base_z

    @property
    def ridge_z_max(self) -> float:
        return self.ridge_base_z + self.ridge_height


@dataclass
class ModelConfig:
    """Neural network architecture parameters."""

    hidden_layers: int = 4
    hidden_width: int = 64
    activation: str = "tanh"
    fourier_features: bool = False
    fourier_scale: float = 1.0
    num_fourier_features: int = 32


@dataclass
class SamplingConfig:
    """Collocation point sampling parameters."""

    n_interior: int = 4096
    n_periodic: int = 512
    n_top: int = 256
    n_bottom: int = 256
    n_interface: int = 256
    sampler_type: str = "sobol"


@dataclass
class TrainingConfig:
    """Training loop parameters."""

    epochs: int = 5000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    use_lbfgs: bool = False
    lbfgs_steps: int = 500
    lbfgs_learning_rate: float = 1.0
    gradient_clip: float = 1.0
    checkpoint_interval: int = 500
    early_stopping_patience: int = 0
    validation_interval: int = 100
    seed: int = 42
    dtype: str = "float64"
    device: str = "auto"
    scheduler: str = "cosine"
    scheduler_step_size: int = 1000
    scheduler_gamma: float = 0.5


@dataclass
class LossWeights:
    """Weights for composite loss terms."""

    pde: float = 1.0
    periodic: float = 10.0
    top: float = 10.0
    bottom: float = 5.0
    interface: float = 0.0
    data: float = 0.0


@dataclass
class PathsConfig:
    """Output paths for artifacts."""

    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    figure_dir: str = "outputs/figures"
    history_file: str = "outputs/training_history.csv"
    results_file: str = "outputs/results.npz"
    metrics_file: str = "outputs/metrics.json"
    config_snapshot: str = "outputs/run_config.yaml"


@dataclass
class PINNConfig:
    """Top-level configuration container."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    paths: PathsConfig = field(default_factory=PathsConfig)
    smoke: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _merge_dataclass(cls: type, data: dict[str, Any] | None) -> Any:
    if not data:
        return cls()
    valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in valid}
    return cls(**filtered)


def load_config(path: str | Path, smoke: bool = False) -> PINNConfig:
    """Load YAML configuration from *path*.

    Parameters
    ----------
    path:
        Path to YAML file.
    smoke:
        If True, apply ``smoke`` overrides for quick test runs.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    config = PINNConfig(
        physics=_merge_dataclass(PhysicsConfig, raw.get("physics")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        sampling=_merge_dataclass(SamplingConfig, raw.get("sampling")),
        training=_merge_dataclass(TrainingConfig, raw.get("training")),
        loss_weights=_merge_dataclass(LossWeights, raw.get("loss_weights")),
        paths=_merge_dataclass(PathsConfig, raw.get("paths")),
        smoke=dict(raw.get("smoke", {})),
        raw=raw,
    )

    if smoke:
        _apply_smoke_overrides(config)

    return config


def _apply_smoke_overrides(config: PINNConfig) -> None:
    """Apply smoke-test overrides from config."""
    smoke = config.smoke
    if not smoke:
        return
    for key, value in smoke.items():
        if hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.sampling, key):
            setattr(config.sampling, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)


def save_config_snapshot(config: PINNConfig, path: str | Path) -> None:
    """Save configuration snapshot to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "physics": config.physics.__dict__,
        "model": config.model.__dict__,
        "sampling": config.sampling.__dict__,
        "training": config.training.__dict__,
        "loss_weights": config.loss_weights.__dict__,
        "paths": config.paths.__dict__,
    }
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(snapshot, fh, default_flow_style=False, sort_keys=False)
