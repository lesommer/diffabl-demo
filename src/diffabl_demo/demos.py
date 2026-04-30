"""Demonstration cases from Lemarié et al. (2021) Section 4."""

import jax
import jax.numpy as jnp
import equinox as eqx
from diffabl_demo.grid import uniform_grid, sinh_stretched_grid
from diffabl_demo.state import ABLState, ABLParams, cbr_params, cch_params
from diffabl_demo.stepper import abl_step, Forcing
from diffabl_demo.solver import run_abl_python


def andren94(
    params: ABLParams | None = None,
    n_steps: int = 1670,
) -> tuple[ABLState, object]:
    g = uniform_grid(41, 1500.0)
    if params is None:
        params = cbr_params(dt=60.0, f=1e-4)

    u_vals = [
        4.44, 4.44, 5.92, 6.91, 7.73, 8.43, 9.02, 9.52, 9.93, 10.25,
        10.47, 10.62, 10.70, 10.71, 10.67, 10.59, 10.48, 10.36, 10.24,
        10.13, 10.04, 9.99, 9.96, 9.95, 9.96, 9.98, 9.99, 10.00, 9.99,
        9.99, 9.99, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00,
        10.00, 10.00, 10.00,
    ]
    v_vals = [
        2.18, 2.18, 2.67, 2.83, 2.84, 2.75, 2.57, 2.34, 2.06, 1.75,
        1.44, 1.12, 0.82, 0.55, 0.31, 0.12, -0.02, -0.11, -0.16, -0.17,
        -0.15, -0.11, -0.06, -0.02, 0.01, 0.02, 0.02, 0.02, 0.02, 0.01,
        0.02, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00,
    ]
    assert len(u_vals) == 41 and len(v_vals) == 41
    u_full = jnp.array(u_vals)
    v_full = jnp.array(v_vals)

    tke_vals = [
        0.0, 0.330, 0.270, 0.225, 0.190, 0.160, 0.133, 0.110, 0.093,
        0.078, 0.063, 0.050, 0.040, 0.030, 0.023, 0.018, 0.013, 0.008,
        0.008, 0.005, 0.005, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
        0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
        0.003, 0.003, 0.003, 0.003, 0.003,
    ]
    assert len(tke_vals) == 41
    tke_full = jnp.array(tke_vals)
    tke_full = tke_full.at[0].set(params.tke_min)
    tke_full = jnp.maximum(tke_full, params.tke_min)

    n = g.jpka
    state = ABLState(
        u=u_full,
        v=v_full,
        theta=jnp.full(n, 273.15),
        q=jnp.full(n, 1e-6),
        tke=tke_full,
        avm=jnp.full(n, params.avm_bak),
        avt=jnp.full(n, params.avt_bak),
        mxlm=jnp.ones(n),
        mxld=jnp.ones(n),
        pblh=jnp.array([1000.0]),
    )

    forcing = Forcing(
        u_geo=10.0,
        v_geo=0.0,
        theta_sfc=288.15,
        q_sfc=1e-6,
    )

    return run_abl_python(state, g, params, forcing, n_steps), g


def cuxart05(
    params: ABLParams | None = None,
    n_steps: int = 3240,
) -> tuple[ABLState, object]:
    g = sinh_stretched_grid(65, zhmax=400.0, zhc=300.0, theta_s=1.0)
    if params is None:
        params = cch_params(dt=10.0, f=1.39e-4)

    n = g.jpka
    theta = jnp.where(g.ght <= 100.0, 265.0, 265.0 + 0.01 * (g.ght - 100.0))
    theta = theta.at[0].set(theta[1])

    state = ABLState(
        u=jnp.full(n, 8.0),
        v=jnp.zeros(n),
        theta=theta,
        q=jnp.full(n, 1e-6),
        tke=jnp.full(n, params.tke_min),
        avm=jnp.full(n, params.avm_bak),
        avt=jnp.full(n, params.avt_bak),
        mxlm=jnp.ones(n),
        mxld=jnp.ones(n),
        pblh=jnp.array([100.0]),
    )

    forcing = Forcing(
        u_geo=8.0,
        v_geo=0.0,
        theta_sfc=256.85,
        q_sfc=1e-6,
    )

    return run_abl_python(state, g, params, forcing, n_steps), g
