"""Tests for nudging, stepper, and solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffabl_demo.grid import uniform_grid
from diffabl_demo.state import ABLState, ABLParams, cbr_params
from diffabl_demo.nudging import nudging_coefficient, apply_nudging
from diffabl_demo.stepper import abl_step, Forcing
from diffabl_demo.solver import run_abl_python


def _make_state_and_grid(n=21, height=500.0):
    g = uniform_grid(n, height)
    state = ABLState(
        u=jnp.full(n, 8.0),
        v=jnp.zeros(n),
        theta=jnp.full(n, 283.0),
        q=jnp.full(n, 1e-6),
        tke=jnp.full(n, 1e-4),
        avm=jnp.full(n, 1e-4),
        avt=jnp.full(n, 1e-5),
        mxlm=jnp.ones(n),
        mxld=jnp.ones(n),
        pblh=jnp.array([500.0]),
    )
    return state, g


def test_nudging_coefficient():
    g = uniform_grid(11, 100.0)
    alpha = nudging_coefficient(g.ght, 100.0, 1e-5, 1e-3)
    assert float(alpha[0]) == pytest.approx(1e-5, rel=1e-3)
    assert float(alpha[-1]) > float(alpha[1])


def test_nudging_relaxation():
    phi = jnp.ones(10) * 2.0
    phi_ref = jnp.ones(10) * 1.0
    alpha = jnp.ones(10) * 1.0
    result = apply_nudging(phi, phi_ref, alpha, dt=1.0)
    assert jnp.all(result < phi)
    assert jnp.all(result > phi_ref)


def test_single_step():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0, f=1e-4)
    forcing = Forcing(u_geo=8.0)
    state_new = abl_step(state, g, params, forcing, step=0)
    assert state_new.u.shape == state.u.shape
    assert jnp.all(jnp.isfinite(state_new.u))


def test_single_step_tke_positive():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0, f=1e-4)
    forcing = Forcing(u_geo=8.0)
    state_new = abl_step(state, g, params, forcing, step=0)
    assert jnp.all(state_new.tke >= params.tke_min)


def test_multi_step_stability():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0, f=1e-4)
    forcing = Forcing(u_geo=8.0)
    for i in range(10):
        state = abl_step(state, g, params, forcing, step=i)
    assert jnp.all(jnp.isfinite(state.u))
    assert jnp.all(jnp.isfinite(state.theta))
    assert jnp.all(state.tke >= params.tke_min)


def test_solver_python():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0, f=1e-4)
    forcing = Forcing(u_geo=8.0)
    state_final = run_abl_python(state, g, params, forcing, n_steps=5)
    assert jnp.all(jnp.isfinite(state_final.u))
