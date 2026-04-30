# diffabl-demo Project Context

Last verified: 2026-04-30

## Purpose
This repository began as a demonstration project for IGE-GeoSciML working group. The purpose of the project is to implement in Jax/Equinox the simplified atmospheric boundary layer (ABL) proposed by Lemarié et al. 2021 (https://doi.org/10.5194/gmd-14-543-2021, pdf available locally in `reference_paper/`) in JAX/Equinox. We also use as reference the FORTRAN implementation of ABL available locally from `nemo_abl1d_GMD_2020/NEMO_CODE/` into a JAX/Equinox/Lineax stack.

The project primary objective is to harden Opencode development skills with a real scientific-computing codebase, testing JAX + Equinox numerics implementation quality and  Opencode context and directive-writing quality.

## Contracts
- **Exposes**: `diffabl-demo` CLI workflows (`munge`, `h2`, `rg`, `h2-cts`, `l2`) and library APIs under `src/diffabl-demo`.
- **Guarantees**: behavior changes are test-backed; docs track current intent and constraints
- **Expects**: contributors optimize for correctness with respect to the reference paper, reproducibility, not feature volume.

## Dependencies
- **Uses**: JAX ecosystem (`jax`, `equinox`, `lineax`, `optimistix`), `scipy`.
- **Used by**: CLI workflows, regression/IO tests, and coding-agent hardening exercises.
- **Boundary**: runtime code in `src/` must not import from `nemo_abl1d_GMD_2020/`.

## Invariants
- `pytest -q` must stay green after behavior changes.
- CLI remains a thin imperative shell around library modules.
- Documentation must explicitly reflect mission or contract shifts.

## Key Decisions
- Treat `docs/diffabl-demo_plan.md` as historical baseline plus dated reassessments.
- Use this root `AGENTS.md` as canonical project context

## Commands
- `pytest -q` - run full test suite.
- `python -m diffable-demo.cli --help` - inspect CLI surface.
- `python -m diffabl-demo.cli <command> --help` - inspect a specific workflow.

## Project Structure
- `src/diffabl-demo/` - package runtime 
- `tests/` - regression, IO, and CLI behavior tests.
- `docs/` - project plans and licensing guidance.

## Behavior Requirements For Agents
- Prefer test-first or test-coupled changes for any behavioral modification.
- Update context docs when project goals, contracts, or boundaries change.
- Report concrete verification evidence (commands and outcomes), not assumptions.

## Boundaries
- Safe to edit: `src/`, `tests/`, `docs/`, `README.md`, `AGENTS.md`.
- Do not copy source text from `nemo_abl1d_GMD_2020` into MIT-licensed runtime code.
