"""Tests for TKE turbulence closure."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffabl_demo.grid import uniform_grid
from diffabl_demo.state import ABLState, ABLParams, cbr_params
from diffabl_demo.tke import abl_zdf_tke, _compute_shear, _compute_N2, _diagnose_pblh


def _make_state_and_grid(n=41, height=1500.0):
    g = uniform_grid(n, height)
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
    return state, g


def test_shear_uniform_wind():
    g = uniform_grid(11, 100.0)
    u = jnp.full(11, 10.0)
    v = jnp.full(11, 0.0)
    S2 = _compute_shear(u, v, g)
    assert jnp.all(S2[2:] < 1e-10)


def test_shear_linear_wind():
    g = uniform_grid(11, 100.0)
    u = jnp.linspace(0, 10, 11)
    v = jnp.zeros(11)
    S2 = _compute_shear(u, v, g)
    assert jnp.all(S2[2:] > 0)


def test_N2_uniform_theta():
    g = uniform_grid(11, 100.0)
    theta = jnp.full(11, 283.0)
    q = jnp.full(11, 1e-6)
    params = cbr_params()
    N2 = _compute_N2(theta, q, g, params)
    assert jnp.allclose(N2[2:], 0.0, atol=1e-4)


def test_tke_positivity():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0)
    ustar2 = 0.1
    tke_new, Km, Kt, l_m, l_d, pblh = abl_zdf_tke(state, g, params, ustar2)
    assert jnp.all(tke_new >= params.tke_min)


def test_tke_Km_Kt_positive():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0)
    ustar2 = 0.1
    tke_new, Km, Kt, l_m, l_d, pblh = abl_zdf_tke(state, g, params, ustar2)
    assert jnp.all(Km >= params.avm_bak)
    assert jnp.all(Kt >= params.avt_bak)


def test_tke_pblh_bounded():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0)
    ustar2 = 0.1
    tke_new, Km, Kt, l_m, l_d, pblh = abl_zdf_tke(state, g, params, ustar2)
    h = float(pblh[0])
    assert h >= float(g.ghw[1])
    assert h <= float(g.ghw[-1])


def test_tke_surface_bc():
    state, g = _make_state_and_grid()
    params = cbr_params(dt=60.0)
    ustar2 = 0.25
    tke_new, Km, Kt, l_m, l_d, pblh = abl_zdf_tke(state, g, params, ustar2)
    e_sfc_expected = ustar2 / jnp.sqrt(params.Cm * params.Ceps)
    assert float(tke_new[0]) == pytest.approx(float(e_sfc_expected), rel=1e-3)
