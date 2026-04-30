"""Tests for implicit vertical diffusion (Thomas solver)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffabl_demo.grid import uniform_grid
from diffabl_demo.diffusion import diffuse_implicit, _thomas_solve


def test_thomas_identity():
    n = 5
    sub = jnp.zeros(n)
    diag = jnp.ones(n)
    sup = jnp.zeros(n)
    rhs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = _thomas_solve(sub, diag, sup, rhs)
    np.testing.assert_allclose(np.array(x), np.array(rhs), atol=1e-6)


def test_thomas_tridiag_simple():
    sub = jnp.array([0.0, -1.0, -1.0, -1.0])
    diag = jnp.array([2.0, 2.0, 2.0, 2.0])
    sup = jnp.array([-1.0, -1.0, -1.0, 0.0])
    rhs = jnp.array([1.0, 0.0, 0.0, 0.0])
    x = _thomas_solve(sub, diag, sup, rhs)
    Ax = diag * x + jnp.concatenate([sup[:-1], jnp.array([0.0])]) * jnp.concatenate([x[1:], jnp.array([0.0])]) + \
         jnp.concatenate([jnp.array([0.0]), sub[1:]]) * jnp.concatenate([jnp.array([0.0]), x[:-1]])
    np.testing.assert_allclose(np.array(Ax), np.array(rhs), atol=1e-6)


def test_diffuse_zero_K():
    g = uniform_grid(41, 1500.0)
    phi = jnp.ones(41)
    K = jnp.zeros(41)
    result = diffuse_implicit(phi, K, g, dt=60.0)
    np.testing.assert_allclose(np.array(result[1:]), np.array(phi[1:]), atol=1e-6)


def test_diffuse_uniform_profile():
    g = uniform_grid(41, 1500.0)
    phi = jnp.ones(41) * 283.0
    K = jnp.ones(41) * 1.0
    result = diffuse_implicit(phi, K, g, dt=60.0)
    np.testing.assert_allclose(np.array(result[1:]), 283.0, atol=0.1)


def test_diffuse_decay():
    g = uniform_grid(41, 1500.0)
    phi = jnp.zeros(41)
    phi = phi.at[20].set(1.0)
    K = jnp.ones(41) * 10.0
    result = diffuse_implicit(phi, K, g, dt=60.0)
    assert jnp.sum(result[1:]) > 0
    assert jnp.all(result[1:] >= 0)


def test_diffuse_differentiable():
    g = uniform_grid(11, 100.0)
    K = jnp.ones(11) * 1.0
    phi = jnp.ones(11) * 283.0

    def loss(K_val):
        result = diffuse_implicit(phi, K_val, g, dt=60.0)
        return jnp.sum(result)

    grad = jax.grad(loss)(K)
    assert jnp.all(jnp.isfinite(grad))
