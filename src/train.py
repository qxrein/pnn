"""Training loop for the grating Helmholtz PINN."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from src.config import save_config_snapshot
from src.losses import compute_losses
from src.model import FieldMLP
from src.sampling import SampleBatch, sample_points
from src.utils import ensure_dirs, resolve_device, resolve_training_dtype, set_seed

if TYPE_CHECKING:
    from src.config import PINNConfig


def _build_scheduler(optimizer: torch.optim.Optimizer, config: PINNConfig):
    sched_name = config.training.scheduler.lower()
    if sched_name == "none":
        return None
    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    if sched_name == "step":
        return StepLR(
            optimizer,
            step_size=config.training.scheduler_step_size,
            gamma=config.training.scheduler_gamma,
        )
    raise ValueError(f"Unknown scheduler: {config.training.scheduler}")


def _losses_to_float(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in loss_dict.items()}


def train_pinn(config: PINNConfig) -> dict[str, Any]:
    """Run full PINN training and return summary metadata."""
    ensure_dirs(config)
    save_config_snapshot(config, config.paths.config_snapshot)

    set_seed(config.training.seed)
    device = resolve_device(config.training.device)
    dtype = resolve_training_dtype(config.training.dtype, device)

    model = FieldMLP(config.model, config.physics).to(device=device, dtype=dtype)
    samples = sample_points(
        config.physics,
        config.sampling,
        device=device,
        dtype=dtype,
        seed=config.training.seed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, config)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, Any] | None = None
    patience_counter = 0
    checkpoint_dir = Path(config.paths.checkpoint_dir)

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_dict = compute_losses(model, samples, config.physics, config.loss_weights)
        total = loss_dict["total"]
        total.backward()

        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        row = {"epoch": epoch, **_losses_to_float(loss_dict)}
        history.append(row)

        # Validation on fresh interior sample every validation_interval
        val_loss = row["total"]
        if epoch % config.training.validation_interval == 0 or epoch == config.training.epochs:
            model.eval()
            with torch.enable_grad():
                val_samples = sample_points(
                    config.physics,
                    config.sampling,
                    device=device,
                    dtype=dtype,
                    seed=config.training.seed + epoch,
                )
                val_dict = compute_losses(model, val_samples, config.physics, config.loss_weights)
                val_loss = float(val_dict["total"].detach().cpu())
            row["val_total"] = val_loss

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "val_loss": best_val,
                        "config": config,
                    },
                    checkpoint_dir / "best_model.pt",
                )
            else:
                patience_counter += 1

        if epoch % config.training.checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                checkpoint_dir / f"checkpoint_epoch_{epoch:06d}.pt",
            )

        if (
            config.training.early_stopping_patience > 0
            and patience_counter >= config.training.early_stopping_patience
        ):
            break

    # LBFGS refinement (optional)
    if config.training.use_lbfgs:
        model.train()
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=config.training.lbfgs_learning_rate,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss_dict = compute_losses(model, samples, config.physics, config.loss_weights)
            loss = loss_dict["total"]
            loss.backward()
            return loss

        for step in range(config.training.lbfgs_steps):
            lbfgs.step(closure)
            loss_dict = compute_losses(model, samples, config.physics, config.loss_weights)
            history.append({"epoch": config.training.epochs + step + 1, **_losses_to_float(loss_dict)})

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        torch.save(
            {"epoch": config.training.epochs, "model_state_dict": model.state_dict(), "config": config},
            checkpoint_dir / "best_model.pt",
        )

    history_path = Path(config.paths.history_file)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)

    return {
        "device": str(device),
        "dtype": str(dtype),
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "history_file": str(history_path),
        "best_checkpoint": str(checkpoint_dir / "best_model.pt"),
    }
