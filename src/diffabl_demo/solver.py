"""Multi-step integration loop for the ABL model."""

import jax
import jax.numpy as jnp
from diffabl_demo.grid import ABLGrid
from diffabl_demo.state import ABLState, ABLParams
from diffabl_demo.stepper import abl_step, Forcing


def run_abl(
    state: ABLState,
    grid: ABLGrid,
    params: ABLParams,
    forcing: Forcing,
    n_steps: int,
) -> ABLState:
    def step_fn(carry, i):
        state = carry
        state = abl_step(state, grid, params, forcing, i)
        return state, None
    state, _ = jax.lax.scan(step_fn, state, jnp.arange(n_steps))
    return state


def run_abl_python(
    state: ABLState,
    grid: ABLGrid,
    params: ABLParams,
    forcing: Forcing,
    n_steps: int,
) -> ABLState:
    for i in range(n_steps):
        state = abl_step(state, grid, params, forcing, i)
    return state
