# Project Status — diffabl-demo

Last updated: 2026-04-30

## Current State

The core ABL solver is **functional and test-backed**. All 48 tests pass. The code is pip-installable and the CLI exposes two demonstration cases.

## Implemented Features

| Module | Status | Notes |
|--------|--------|-------|
| Vertical grid (uniform, sinh-stretched) | Done | Matching NEMO ABL convention |
| ABLState / ABLParams | Done | CBR and CCH parameter sets |
| Implicit vertical diffusion (Thomas) | Done | `jax.lax.scan`-based, JIT-compatible |
| TKE closure | Done | Deardorff mixing length, Patankar positivity |
| PBL height diagnosis | Done | Bulk Richardson number criterion |
| Coriolis (forward-backward, semi-implicit) | Done | Alternating scheme for stability |
| Surface BCs (bulk formulae) | Done | Simplified Cd/Ch/Ce |
| Nudging | Done | Height-dependent cubic profile |
| Full stepper | Done | Assembles all components |
| Multi-step solver | Done | Python loop (JAX scan pending) |
| Differentiability | Verified | `jax.grad` and finite-difference agreement |
| CLI | Done | `andren94` and `cuxart05` subcommands |
| Demo: Andren 1994 (Fig. 4) | Partial | Running, results not yet validated against published |
| Demo: Cuxart 2005 (Fig. 5) | Partial | Running, results not yet validated against published |
| Demo: 2D x-z (Figs. 6-7) | Not started | |
| Demo: SCM t-z (Figs. 8-9) | Not started | Requires MESONH forcing data |
| Mixing length: BL89 (nn_amxl=2) | Not started | |
| Mixing length: Modified BL89 (nn_amxl=3) | Not started | |
| COARE/ECMWF bulk algorithms | Not started | |
| JIT compilation | Partial | Thomas solver uses `lax.scan`; TKE mixing-length loops use Python for-loops |
| 2D vmap | Not started | Currently single-column only |

## Known Limitations

1. Mixing-length sweep uses Python for-loops (not JIT-friendly for variable grid sizes)
2. Only Deardorff (nn_amxl=0) mixing length implemented; BL89 and modified BL89 pending
3. No MESONH forcing integration for SCM experiments
4. No visualization/plotting utilities yet
5. Bulk formulae are simplified (constant-like Cd); COARE/ECMWF not implemented
6. `run_abl` with `jax.lax.scan` not tested (only `run_abl_python` used)

## Next Steps

1. Implement BL89 and modified BL89 mixing lengths
2. Validate Andren94 and Cuxart05 results against published figures
3. Add COARE 3.0 / ECMWF bulk algorithms
4. Implement 2D domain support via `jax.vmap`
5. Add plotting utilities for figure reproduction
6. Make mixing-length sweeps JIT-compatible via `jax.lax.scan`
