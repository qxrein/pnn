# PINN Optics — 2D Grating Helmholtz PINN Prototype

**Research project:** *Learning the Electromagnetic-to-Optical-System Transfer Function with Physics-Informed Neural Networks*

This repository contains the **first milestone**: a proof-of-concept Physics-Informed Neural Network (PINN) that solves 2D frequency-domain electromagnetic scattering from a binary dielectric diffraction grating using a **scalar Helmholtz formulation**.

> **Important:** This is a first 2D scalar Helmholtz PINN research prototype. It is **not** a production electromagnetic solver and **not** yet a complete vector Maxwell PINN. Physical validation requires comparison against an independent reference solver (RCWA, FDTD, FEM, etc.).

---

## Research objective

Build and validate a modular PINN framework for grating scattering before extending to:

1. Full-vector Maxwell equations in 2D/3D
2. Import and comparison against industry EM solver reference data
3. Learning an electromagnetic-to-optical-system transfer function
4. Inverse design and optimization

---

## Physical problem

Time-harmonic scalar field \(E(x,z)\) satisfies the 2D Helmholtz equation:

\[
\frac{\partial^2 E}{\partial x^2} + \frac{\partial^2 E}{\partial z^2} + k_0^2 \,\varepsilon_r(x,z)\, E = 0
\]

where \(k_0 = 2\pi/\lambda\) and \(\varepsilon_r(x,z)\) is the piecewise-constant relative permittivity of a binary grating (air / ridge / substrate).

The complex field is represented as two real network outputs: `E_real`, `E_imag`.

### Coordinate convention

| Axis | Range | Meaning |
|------|-------|---------|
| **x** | `[0, period]` | Periodic horizontal direction (one grating cell) |
| **z** | `[0, domain_height]` | Vertical direction; **z = 0 is top** (incident boundary), **z increases downward** |

### Boundary conditions (prototype)

| Boundary | Treatment |
|----------|-----------|
| Left / right (x) | Periodic: \(E(0,z) = E(\Lambda,z)\) |
| Top (z = 0) | Soft Dirichlet: \(E \approx e^{-ik_0 z}\) (normal incidence, downward propagation) |
| Bottom | Approximate outgoing-wave soft target (documented approximation, not rigorous ABC) |

### Known limitations

- Scalar TE/TM reduction — not full vector Maxwell
- Absorbing boundaries are approximate
- Sharp εᵣ discontinuities can impede PINN convergence; interior points exclude a margin around interfaces by default
- Results are **not physically validated** until reference-solver comparison is performed
- MPS (Apple Metal) uses float32 automatically for compatibility

---

## Folder structure

```
pinn_optics/
├── README.md
├── requirements.txt
├── configs/default.yaml
├── src/
│   ├── config.py           # Dataclasses + YAML loading
│   ├── geometry.py         # εᵣ(x,z) and coordinate normalization
│   ├── sampling.py         # Collocation point sampling (uniform / Sobol)
│   ├── model.py            # MLP field network
│   ├── derivatives.py      # Autograd spatial derivatives
│   ├── physics.py          # Helmholtz residual
│   ├── boundary_conditions.py
│   ├── losses.py           # Composite loss
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Grid evaluation + metrics
│   ├── visualization.py    # Matplotlib figures
│   ├── reference_data.py   # External solver data import (placeholder workflow)
│   └── utils.py            # Device (CPU/CUDA/MPS), seeding
├── scripts/
│   ├── train_pinn.py
│   ├── evaluate_pinn.py
│   └── plot_results.py
├── tests/
└── outputs/
```

---

## Installation

```bash
cd pinn_optics
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

Requires **Python 3.10+**, **PyTorch 2.0+** (with MPS support on Apple Silicon).

---

## Training

```bash
python scripts/train_pinn.py --config configs/default.yaml
```

Quick smoke test (50 epochs, smaller network):

```bash
python scripts/train_pinn.py --config configs/default.yaml --smoke
```

Outputs:
- `outputs/checkpoints/best_model.pt`
- `outputs/training_history.csv`
- `outputs/run_config.yaml`

### Device selection

Set in `configs/default.yaml`:

```yaml
training:
  device: auto   # auto | cpu | cuda | mps
```

`auto` selects CUDA → MPS → CPU. On macOS Apple Silicon, **MPS (Metal)** is used automatically.

---

## Evaluation

```bash
python scripts/evaluate_pinn.py \
  --config configs/default.yaml \
  --checkpoint outputs/checkpoints/best_model.pt
```

Outputs:
- `outputs/results.npz` — x, z, E_real, E_imag, |E|, |E|², phase
- `outputs/metrics.json` — PDE MSE, periodic BC error, etc.

---

## Figures

```bash
python scripts/plot_results.py \
  --results outputs/results.npz \
  --checkpoint outputs/checkpoints/best_model.pt \
  --history outputs/training_history.csv
```

Generates publication-quality PNG (600 dpi) and PDF figures in `outputs/figures/`.

---

## Independent Reference Validation

Once you have field data from an external electromagnetic solver (e.g., RCWA via S4, FDTD via MEEP, or FEM via COMSOL), you can run a quantitative comparison against the trained PINN.

> **Important:** No fake or synthetic reference data are generated automatically.
> The comparison is scientifically meaningful only when the reference originates
> from an independent, validated solver.

### Expected NPZ format

Export your solver's field data as a NumPy NPZ file with **exactly** these arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `x` | `(Nx,)` | 1-D horizontal coordinate array |
| `z` | `(Nz,)` | 1-D vertical coordinate array |
| `E_real` | `(Nz, Nx)` | Real part of the complex field |
| `E_imag` | `(Nz, Nx)` | Imaginary part of the complex field |

Field indexing convention:

```
E(z_index, x_index) = E_real[z_index, x_index] + 1j * E_imag[z_index, x_index]
```

- The **first** field dimension corresponds to **z** (rows).
- The **second** field dimension corresponds to **x** (columns).
- Coordinate arrays must be finite, 1-D, and strictly monotonic (increasing or decreasing).
- Decreasing coordinates are normalised to increasing order internally; field axes are reversed consistently.

Create the file from Python:

```python
import numpy as np
np.savez("reference_grating.npz",
         x=x_array,       # shape (Nx,)
         z=z_array,       # shape (Nz,)
         E_real=E_real,   # shape (Nz, Nx)
         E_imag=E_imag)   # shape (Nz, Nx)
```

### Running reference validation

```bash
python scripts/evaluate_pinn.py \
    --config configs/default.yaml \
    --checkpoint outputs/checkpoints/best_model.pt \
    --reference path/to/reference.npz
```

If `--reference` is omitted, reference validation is skipped and a note is printed. If the specified file does not exist, the following message is printed and evaluation completes normally (exit code 0):

```
No reference file found. Skipping reference validation.
Provide an NPZ file containing x, z, E_real, and E_imag.
```

### Output files

| File | Description |
|------|-------------|
| `outputs/figures/reference_metrics.json` | Four comparison metrics |
| `outputs/figures/reference_comparison.png` | Three-panel figure (600 dpi) |
| `outputs/figures/reference_comparison.pdf` | Vector figure |

### Reported metrics

| Metric | Description |
|--------|-------------|
| `relative_l2_error_real` | Relative L2 error of Re{E} |
| `relative_l2_error_imag` | Relative L2 error of Im{E} |
| `relative_l2_error_magnitude` | Relative L2 error of \|E\| |
| `maximum_absolute_error` | Max pointwise \|PINN_mag − ref_mag\| |

The reference field is interpolated onto the PINN evaluation grid before comparison. Only points where both fields are finite are included in the metrics.

### Using the API directly

```python
from src.reference_data import run_comparison

metrics = run_comparison(
    reference_path="reference_grating.npz",
    pinn_x=pinn_x_1d,           # shape (Nx,)
    pinn_z=pinn_z_1d,           # shape (Nz,)
    pinn_E_real=pinn_E_real,    # shape (Nz, Nx)
    pinn_E_imag=pinn_E_imag,    # shape (Nz, Nx)
    output_dir="outputs/figures",
    save_figure=True,
)
if metrics is not None:
    print(metrics["relative_l2_error_magnitude"])
```

Enable supervised data loss during training by providing reference samples and setting `loss_weights.data > 0` in `configs/default.yaml`.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Next research milestones

1. **Reference validation** — RCWA/FDTD export pipeline and quantitative error metrics
2. **Improved BCs** — UPML, DtN, or scattered-field formulation
3. **Vector Maxwell PINN** — full \( \mathbf{E} \), \( \mathbf{H} \) coupling
4. **Transfer function stage** — map EM response features → optical system metrics
5. **Inverse design** — gradient-based grating optimization via PINN surrogate

---

## License

Research prototype — adapt as needed for academic use.
