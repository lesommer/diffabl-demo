"""Tests for Coriolis treatment."""

import jax.numpy as jnp
import numpy as np
import pytest
from diffabl_demo.coriolis import coriolis_forward_backward, coriolis_semi_implicit


def test_forward_backward_no_rotation():
    u = jnp.ones(10) * 10.0
    v = jnp.zeros(10)
    u_new, v_new = coriolis_forward_backward(u, v, f=0.0, dt=60.0, step=0)
    np.testing.assert_allclose(np.array(u_new), 10.0, atol=1e-10)
    np.testing.assert_allclose(np.array(v_new), 0.0, atol=1e-10)


def test_forward_backward_alternation():
    u = jnp.ones(5) * 10.0
    v = jnp.zeros(5)
    f = 1e-4
    dt = 60.0
    u0, v0 = coriolis_forward_backward(u, v, f, dt, step=0)
    u1, v1 = coriolis_forward_backward(u0, v0, f, dt, step=1)
    assert not jnp.allclose(u0, u1)


def test_semi_implicit_geostrophic():
    u = jnp.zeros(10)
    v = jnp.zeros(10)
    u_geo = 10.0
    v_geo = 0.0
    f = 1e-4
    dt = 60.0
    u_new, v_new = coriolis_semi_implicit(u, v, f, dt, u_geo, v_geo)
    assert jnp.mean(jnp.abs(u_new)) > 0


def test_semi_implicit_no_coriolis():
    u = jnp.ones(5) * 5.0
    v = jnp.ones(5) * 3.0
    u_new, v_new = coriolis_semi_implicit(u, v, f=0.0, dt=60.0)
    np.testing.assert_allclose(np.array(u_new), 5.0, atol=1e-10)
    np.testing.assert_allclose(np.array(v_new), 3.0, atol=1e-10)
