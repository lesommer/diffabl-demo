"""Tests verifying differentiability of the ABL solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
from diffabl_demo.grid import uniform_grid
from diffabl_demo.state import ABLState, ABLParams, cbr_params
from diffabl_demo.stepper import abl_step, Forcing
from diffabl_demo.solver import run_abl_python


def _make_state_and_grid(n=11, height=200.0):
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
        pblh=jnp.array([200.0]),
    )
    return state, g


def test_grad_through_single_step():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0, f=1e-4)
    forcing = Forcing(u_geo=8.0)

    def loss_fn(u0_val):
        s = eqx.tree_at(lambda s: s.u, state, jnp.full(g.jpka, u0_val))
        s_new = abl_step(s, g, params, forcing, 0)
        return jnp.sum(s_new.u ** 2)

    grad = jax.grad(loss_fn)(8.0)
    assert jnp.isfinite(grad)
    assert grad != 0.0


def test_grad_through_multi_step():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0, f=1e-4)
    forcing = Forcing(u_geo=8.0)

    def loss_fn(theta_val):
        s = eqx.tree_at(lambda s: s.theta, state, jnp.full(g.jpka, theta_val))
        s_final = run_abl_python(s, g, params, forcing, 3)
        return jnp.sum(s_final.theta ** 2)

    grad = jax.grad(loss_fn)(283.0)
    assert jnp.isfinite(grad)


def test_grad_wrt_theta_sfc():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0, f=1e-4)

    def loss_fn(theta_sfc):
        f = Forcing(u_geo=8.0, theta_sfc=theta_sfc)
        s_new = abl_step(state, g, params, f, 0)
        return jnp.sum(s_new.theta ** 2)

    grad = jax.grad(loss_fn)(283.0)
    assert jnp.isfinite(grad)


def test_grad_finite_diff_agreement():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0, f=1e-4)
    forcing = Forcing(u_geo=8.0)

    def loss_fn(u0_val):
        s = eqx.tree_at(lambda s: s.u, state, jnp.full(g.jpka, u0_val))
        s_new = abl_step(s, g, params, forcing, 0)
        return jnp.sum(s_new.u[1:] ** 2)

    u0 = 8.0
    eps = 1e-4
    l_plus = loss_fn(u0 + eps)
    l_minus = loss_fn(u0 - eps)
    fd_grad = (l_plus - l_minus) / (2 * eps)

    ad_grad = jax.grad(loss_fn)(u0)

    np.testing.assert_allclose(float(ad_grad), float(fd_grad), rtol=0.05)
