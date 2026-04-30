# diffabl-demo

A differentiable simplified atmospheric boundary layer (ABL) model implemented in JAX/Equinox, following the formulation of Lemarié et al. (2021).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## CLI Usage

```bash
# Andren 1994 Ekman spiral (Fig. 4)
diffabl-demo andren94 --steps 1670 --params cbr

# Cuxart 2005 convective ABL (Fig. 5)
diffabl-demo cuxart05 --steps 3240 --params cch
```

## Library API

```python
from diffabl_demo.demos import andren94, cuxart05
from diffabl_demo.state import cbr_params, cch_params

# Run Andren 1994 with CBR parameters
state, grid = andren94(cbr_params(dt=60.0, f=1e-4), n_steps=1670)

# Run Cuxart 2005 with CCH/MesoNH parameters
state, grid = cuxart05(cch_params(dt=10.0, f=1.39e-4), n_steps=3240)
```

## Algorithm

The solver implements a 1.5-order TKE (turbulent kinetic energy) closure on a 1D staggered vertical grid:

1. **TKE equation** — implicit Euler-backward with Patankar positivity treatment
2. **Mixing length** — Deardorff (1980) with upward/downward sweeps
3. **PBL height** — diagnosed from bulk Richardson number integral
4. **Tracer diffusion** — implicit Euler-backward with Robin surface BCs
5. **Coriolis** — forward-backward alternating scheme
6. **Momentum diffusion** — implicit Euler-backward with surface drag BC
7. **Nudging** — optional height-dependent Newtonian relaxation

See `docs/continuous_equations.md` and `docs/discrete_implementation.md` for full mathematical details.

## Differentiability

The entire solver is differentiable in both forward and reverse mode via JAX autodiff:

```python
import jax
from diffabl_demo.stepper import abl_step, Forcing

def loss(u0):
    state = ...  # initial state with u=jnp.full(n, u0)
    state_new = abl_step(state, grid, params, forcing, 0)
    return jnp.sum(state_new.u ** 2)

grad_u0 = jax.grad(loss)(8.0)  # reverse-mode AD
```

## Testing

```bash
pytest -q   # 48 tests covering all modules
```

## Project Structure

```
src/diffabl_demo/
  grid.py          Vertical grid (uniform, sinh-stretched)
  state.py         ABLState, ABLParams (CBR and CCH parameter sets)
  diffusion.py     Implicit vertical diffusion via Thomas algorithm
  tke.py           TKE turbulence closure
  coriolis.py      Coriolis treatment (forward-backward, semi-implicit)
  boundary.py      Bulk formulae (simplified Cd, Ch, Ce)
  nudging.py       Height-dependent Newtonian relaxation
  stepper.py       Single time-step driver
  solver.py        Multi-step integration loop
  demos.py         Andren94 and Cuxart05 demonstration cases
  cli.py           Command-line interface
tests/
  test_grid.py, test_diffusion.py, test_tke.py, test_coriolis.py,
  test_boundary.py, test_stepper.py, test_demos.py, test_differentiability.py
docs/
  implementation_plan.md, continuous_equations.md, discrete_implementation.md
```

## Reference

Lemarié, F., Samson, G., Redelsperger, J.-L., Giordani, H., Brivoal, T., & Madec, G. (2021).
A simplified atmospheric boundary layer model for an improved representation of air–sea interactions in eddying oceanic models.
*Geoscientific Model Development*, 14, 543–572. https://doi.org/10.5194/gmd-14-543-2021

## License

MIT
