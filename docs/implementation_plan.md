# Implementation Plan — diffabl-demo

Last updated: 2026-04-30

## Objective

Implement the simplified atmospheric boundary layer (ABL) model of Lemarié et al. (2021) in JAX/Equinox, ensuring forward- and reverse-mode autodiff through the entire solver, with test-backed correctness and reproducible demonstration cases matching Section 4 of the paper.

## Reference Sources

| Source | Location | Role |
|--------|----------|------|
| Lemarié et al. 2021 (GMD 14, 543–572) | `reference_paper/yW8749-gmd-14-543-2021.pdf` | Equations, physics, test cases |
| NEMO ABL FORTRAN code | `nemo_abl1d_GMD_2020/NEMO_CODE/src/ABL/` | Reference implementation |
| MESONH output data | `nemo_abl1d_GMD_2020/MESONH_output/` | Forcing for SCM experiments |
| Idealized configs | `nemo_abl1d_GMD_2020/NEMO_CODE/cfgs/` | Test case parameters |

## Architecture

```
src/diffabl_demo/
├── __init__.py
├── grid.py          # Vertical grid construction (uniform, sinh-stretched)
├── state.py         # ABLState dataclass (u, v, theta, q, tke + diagnostics)
├── params.py        # Physical and numerical parameters (equinox.Module)
├── tke.py           # TKE turbulence closure (mixing length, Km, Kt, PBL height)
├── diffusion.py     # Implicit vertical diffusion (Thomas algorithm)
├── coriolis.py      # Coriolis treatment (forward-backward, semi-implicit)
├── boundary.py      # Surface/toplevel boundary conditions (bulk formulae)
├── nudging.py       # Height-dependent Newtonian relaxation
├── stepper.py       # Main time-step driver (assembles all pieces)
├── solver.py        # Multi-step integration loop
└── cli.py           # Command-line interface
tests/
├── test_grid.py
├── test_diffusion.py
├── test_tke.py
├── test_coriolis.py
├── test_boundary.py
├── test_nudging.py
├── test_stepper.py
├── test_solver.py
├── test_differentiability.py
├── test_andren94.py
├── test_cuxart05.py
└── test_scm.py
```

## Phased Implementation

### Phase 0 — Project Skeleton

- `pyproject.toml` with JAX, equinox, lineax, optimistix, scipy deps
- Package structure under `src/diffabl_demo/`
- `pytest` configuration
- `pip install -e .` must work

### Phase 1 — Grid, State, Parameters

- `grid.py`: build vertical grid (uniform + sinh-stretched); compute `ght`, `ghw`, `e3t`, `e3w`
- `state.py`: `ABLState` equinox dataclass with `u`, `v`, `tq`, `tke`, diagnostic arrays
- `params.py`: `ABLParams` equinox module with two parameter sets (CBR, CCH/MesoNH) and namelist-like defaults
- **Tests**: grid properties (sum of e3t ≈ domain height), state creation, parameter set switching

### Phase 2 — Implicit Vertical Diffusion (Thomas Solver)

- `diffusion.py`: tridiagonal solver via `lineax` or manual Thomas algorithm, vectorised over columns
- Build LHS/RHS for generic diffusion equation with Robin/Neumann BCs
- **Tests**: known analytical solutions (pure diffusion decay), conservation, tridiagonal solve correctness

### Phase 3 — TKE Turbulence Closure

- `tke.py`: full TKE equation (shear production, buoyancy, dissipation, vertical diffusion)
  - Patankar trick for positivity
  - 4 mixing-length options (Deardorff, Modified Deardorff, BL89, Modified BL89)
  - Stability function φ_z and Km, Kt diagnostics
  - PBL height diagnosis via bulk Richardson number
- **Tests**: TKE positivity, mixing-length boundedness, Km/Kt physical ranges, PBL height convergence

### Phase 4 — Coriolis and Momentum

- `coriolis.py`: forward-backward scheme + semi-implicit with geostrophic guide
- Assemble momentum diffusion tridiagonal with surface drag BC
- **Tests**: geostrophic balance, Ekman spiral shape, Coriolis stability

### Phase 5 — Tracer Equations

- `boundary.py`: bulk formulae (Cd, Ch, Ce from NCAR/COARE/ECMWF)
- Implicit diffusion for θ and q with Robin surface BCs
- **Tests**: bulk formula coefficients, surface flux signs, tracer conservation

### Phase 6 — Nudging and Full Stepper

- `nudging.py`: height-dependent relaxation profile (cubic in σ = z/h_pbl)
- `stepper.py`: single time-step assembling TKE → tracers → Coriolis → momentum → nudging
- `solver.py`: multi-step loop with time-index swapping
- **Tests**: single-step forward, multi-step stability, relaxation toward reference

### Phase 7 — Demonstration Cases (Paper Section 4)

| Case | Paper Figure | Config | Key verification |
|------|-------------|--------|-------------------|
| Andren 1994 Ekman spiral | Fig. 4 | `ABL_IDEAL_ANDREN94` | Wind hodograph matches published |
| Cuxart 2005 convective ABL | Fig. 5 | `ABL_IDEAL_CUXART05` | Potential temperature profile, TKE profile |
| 2D x-z cross-sections | Figs. 6–7 | `ABL_IDEAL_2DXZ` | Spatial structure of ABL fields |
| SCM time-z Hovmöller | Figs. 8–9 | `ABL_IDEAL_2DTZ` | Time-height diagrams vs MESONH |

- Implement each case as a function that sets up grid, ICs, forcing, and runs the solver
- Reproduce published plots (matplotlib)

### Phase 8 — Differentiability Verification

- `test_differentiability.py`: verify `jax.grad` and `jax.jacfwd` through full solver
- Check gradients against finite differences
- Profile computational cost of backward pass

### Phase 9 — CLI and Documentation

- CLI exposing all demo cases
- README.md updated with usage, algorithm description, project status
- `docs/project_status.md`

## Branching Strategy

- `main` — always pip-installable, tests green
- `feat/phase-N` — development branches, merged to main when phase complete and tests pass
- Squash-merge to keep main history clean

## Quality Gates (per phase)

1. `pip install -e .` succeeds
2. `pytest -q` passes (all existing + new tests)
3. `docs/project_status.md` updated
4. `README.md` reflects current CLI and capabilities
