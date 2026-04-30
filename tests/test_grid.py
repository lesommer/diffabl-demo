"""Tests for grid construction and ABL state/params."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffabl_demo.grid import uniform_grid, sinh_stretched_grid
from diffabl_demo.state import ABLState, ABLParams, cbr_params, cch_params


def test_uniform_grid_shape():
    g = uniform_grid(41, 1500.0)
    assert g.ght.shape == (41,)
    assert g.ghw.shape == (41,)
    assert g.e3t.shape == (41,)
    assert g.e3w.shape == (41,)
    assert g.jpka == 41
    assert g.jpkam1 == 40


def test_uniform_grid_spacing():
    g = uniform_grid(41, 1500.0)
    np.testing.assert_allclose(np.array(g.e3t[1:]), 37.5, rtol=1e-6)


def test_uniform_grid_height():
    g = uniform_grid(41, 1500.0)
    assert g.height == pytest.approx(1500.0, rel=1e-5)


def test_uniform_grid_surface():
    g = uniform_grid(41, 1500.0)
    assert float(g.ghw[0]) == pytest.approx(0.0)
    assert float(g.ght[0]) == pytest.approx(0.0)
    assert float(g.e3t[0]) == pytest.approx(0.0)


def test_sinh_stretched_grid_shape():
    g = sinh_stretched_grid(65, zhmax=400.0, zhc=300.0, theta_s=1.0)
    assert g.ght.shape == (65,)
    assert g.jpka == 65


def test_sinh_stretched_grid_monotonic():
    g = sinh_stretched_grid(65, zhmax=400.0, zhc=300.0, theta_s=1.0)
    assert jnp.all(jnp.diff(g.ght[1:]) > 0)


def test_sinh_stretched_grid_positivity():
    g = sinh_stretched_grid(65, zhmax=400.0, zhc=300.0, theta_s=1.0)
    assert jnp.all(g.e3t[1:] > 0)
    assert jnp.all(g.e3w > 0)


def test_sinh_stretched_grid_surface():
    g = sinh_stretched_grid(65, zhmax=400.0, zhc=300.0, theta_s=1.0)
    assert float(g.ghw[0]) == pytest.approx(0.0)
    assert float(g.ght[0]) == pytest.approx(0.0)
    assert float(g.e3t[0]) == pytest.approx(0.0)


def test_cbr_params():
    p = cbr_params()
    assert p.Cm == 0.0667
    assert p.Ceps == 0.7
    assert p.Ric == 0.139


def test_cch_params():
    p = cch_params()
    assert p.Cm == 0.126
    assert p.Ceps == 0.845
    assert p.Ric == 0.143


def test_params_override():
    p = cbr_params(Cm=0.1)
    assert p.Cm == 0.1
    assert p.Ct == 0.1667


def test_abl_state_creation():
    n = 41
    state = ABLState(
        u=jnp.zeros(n),
        v=jnp.zeros(n),
        theta=jnp.full(n, 283.0),
        q=jnp.full(n, 1e-6),
        tke=jnp.full(n, 1e-4),
        avm=jnp.full(n, 1e-4),
        avt=jnp.full(n, 1e-5),
        mxlm=jnp.ones(n),
        mxld=jnp.ones(n),
        pblh=jnp.array([1000.0]),
    )
    assert state.u.shape == (n,)
    assert state.pblh.shape == (1,)
