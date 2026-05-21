# Project Status — diffabl-demo

Last updated: 2026-05-21

## Current State

The core ABL solver is **functional and test-backed**. All 57 tests pass (48 existing + 9 new BL89 tests). The code is pip-installable and the CLI exposes two demonstration cases.

## Implemented Features

| Module | Status | Notes |
|--------|--------|-------|
| Vertical grid (uniform, sinh-stretched) | Done | Matching NEMO ABL convention |
| ABLState / ABLParams | Done | CBR and CCH parameter sets; derived params (mxl_min, rn_Lsfc, rn_Esfc) |
| Implicit vertical diffusion (Thomas) | Done | `jax.lax.scan`-based, JIT-compatible |
| TKE closure | Done | Deardorff + BL89 + Modified BL89 mixing lengths, Patankar positivity |
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
| Mixing length: BL89 (nn_amxl=2) | Done | `lax.scan` + `vmap` vectorized integral sign-change search |
| Mixing length: Modified BL89 (nn_amxl=3) | Done | BL89 + Rod-weighted shear production in integral |
| COARE/ECMWF bulk algorithms | Not started | |
| JIT compilation | Partial | Thomas solver uses `lax.scan`; BL89 search uses `lax.scan` + `vmap`; Deardorff sweep uses `lax.scan` |
| 2D vmap | Not started | Currently single-column only |

## Known Limitations

1. Only Deardorff (nn_amxl=0/1) and BL89/Modified BL89 (nn_amxl=2/3) mixing lengths; no "Modified Deardorff" variant
2. No MESONH forcing integration for SCM experiments
3. Plotting utilities not yet integrated into CLI or tested
4. Bulk formulae are simplified (constant-like Cd); COARE/ECMWF not implemented
5. `run_abl` with `jax.lax.scan` not tested (only `run_abl_python` used)
6. BL89 search O(n^2) via vmap; could be optimized with cumulative-sum decomposition

## Next Steps

1. Validate Andren94 and Cuxart05 results against published figures
2. Add COARE 3.0 / ECMWF bulk algorithms
3. Implement 2D domain support via `jax.vmap`
4. Optimize BL89 search with cumulative-sum decomposition for O(n) per level
5. Add CLI flag and tests for plotting utilities
