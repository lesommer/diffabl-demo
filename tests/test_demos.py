"""Tests for demo cases (Andren 94 and Cuxart 05)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffabl_demo.state import cbr_params, cch_params
from diffabl_demo.demos import andren94, cuxart05


def test_andren94_short():
    params = cbr_params(dt=60.0, f=1e-4)
    state, g = andren94(params, n_steps=10)
    assert jnp.all(jnp.isfinite(state.u))
    assert jnp.all(jnp.isfinite(state.v))
    assert jnp.all(jnp.isfinite(state.theta))
    assert jnp.all(state.tke >= params.tke_min)


def test_andren94_wind_remains_bounded():
    params = cbr_params(dt=60.0, f=1e-4)
    state, g = andren94(params, n_steps=50)
    assert jnp.all(jnp.abs(state.u) < 30.0)
    assert jnp.all(jnp.abs(state.v) < 30.0)


def test_cuxart05_short():
    params = cch_params(dt=10.0, f=1.39e-4)
    state, g = cuxart05(params, n_steps=10)
    assert jnp.all(jnp.isfinite(state.u))
    assert jnp.all(jnp.isfinite(state.theta))
    assert jnp.all(state.tke >= params.tke_min)


def test_cuxart05_theta_bounded():
    params = cch_params(dt=10.0, f=1.39e-4)
    state, g = cuxart05(params, n_steps=50)
    assert jnp.all(state.theta[1:] > 200.0)
    assert jnp.all(state.theta[1:] < 400.0)
