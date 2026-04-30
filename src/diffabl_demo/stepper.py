"""Main time-step driver for the ABL model.

Assembles TKE closure, tracer diffusion, Coriolis, momentum diffusion,
and nudging into a single Euler-backward time step. Fully JAX-autodiff
compatible (no Python float/int on traced values, no Python control flow
on traced predicates).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from diffabl_demo.grid import ABLGrid
from diffabl_demo.state import ABLState, ABLParams
from diffabl_demo.tke import abl_zdf_tke
from diffabl_demo.diffusion import diffuse_implicit
from diffabl_demo.coriolis import apply_coriolis
from diffabl_demo.boundary import bulk_Cd, bulk_Ch, bulk_Ce
from diffabl_demo.nudging import nudging_coefficient, apply_nudging


class Forcing(eqx.Module):
    u_ocean: float = 0.0
    v_ocean: float = 0.0
    theta_sfc: float = 283.0
    q_sfc: float = 1e-6
    u_geo: float = 0.0
    v_geo: float = 0.0
    theta_ref: jnp.ndarray | None = None
    q_ref: jnp.ndarray | None = None
    u_ref: jnp.ndarray | None = None
    v_ref: jnp.ndarray | None = None
    alpha_tra_min: float = 0.0
    alpha_tra_max: float = 0.0
    alpha_dyn_min: float = 0.0
    alpha_dyn_max: float = 0.0


def abl_step(
    state: ABLState,
    grid: ABLGrid,
    params: ABLParams,
    forcing: Forcing,
    step: int,
) -> ABLState:
    dt = params.dt
    n = grid.jpka

    u = state.u
    v = state.v
    theta = state.theta
    q = state.q
    tke = state.tke

    wind_speed = jnp.sqrt(u[1] ** 2 + v[1] ** 2 + 1e-10)
    Cd = bulk_Cd(wind_speed)
    Ch = bulk_Ch(wind_speed)
    Ce = bulk_Ce(wind_speed)
    ustar2 = Cd * ((u[1] - forcing.u_ocean) ** 2 + (v[1] - forcing.v_ocean) ** 2)

    tke_new, Km, Kt, mxlm, mxld, pblh = abl_zdf_tke(state, grid, params, ustar2)

    sfc_heat = dt * Ch * wind_speed
    sfc_moist = dt * Ce * wind_speed
    sfc_heat_rhs = dt * Ch * wind_speed * forcing.theta_sfc
    sfc_moist_rhs = dt * Ce * wind_speed * forcing.q_sfc

    theta_new = diffuse_implicit(theta, Kt, grid, dt,
                                  surface_coeff=sfc_heat,
                                  surface_rhs=sfc_heat_rhs)
    q_new = diffuse_implicit(q, Kt, grid, dt,
                              surface_coeff=sfc_moist,
                              surface_rhs=sfc_moist_rhs)

    u, v = apply_coriolis(u, v, params, step,
                          forcing.u_geo, forcing.v_geo)

    sfc_mom = dt * Cd * wind_speed
    sfc_mom_rhs_u = dt * Cd * wind_speed * forcing.u_ocean
    sfc_mom_rhs_v = dt * Cd * wind_speed * forcing.v_ocean

    u_new = diffuse_implicit(u, Km, grid, dt,
                              surface_coeff=sfc_mom,
                              surface_rhs=sfc_mom_rhs_u)
    v_new = diffuse_implicit(v, Km, grid, dt,
                              surface_coeff=sfc_mom,
                              surface_rhs=sfc_mom_rhs_v)

    pblh_val = pblh[0]
    do_nudge_tra = (forcing.alpha_tra_min > 0) and (forcing.theta_ref is not None)
    do_nudge_dyn = (forcing.alpha_dyn_min > 0) and (forcing.u_ref is not None)

    if do_nudge_tra:
        alpha = nudging_coefficient(grid.ght, pblh_val,
                                     forcing.alpha_tra_min, forcing.alpha_tra_max)
        theta_new = apply_nudging(theta_new, forcing.theta_ref, alpha, dt)
        q_new = apply_nudging(q_new, forcing.q_ref, alpha, dt)

    if do_nudge_dyn:
        alpha_d = nudging_coefficient(grid.ght, pblh_val,
                                       forcing.alpha_dyn_min, forcing.alpha_dyn_max)
        u_new = apply_nudging(u_new, forcing.u_ref, alpha_d, dt)
        v_new = apply_nudging(v_new, forcing.v_ref, alpha_d, dt)

    return ABLState(
        u=u_new,
        v=v_new,
        theta=theta_new,
        q=q_new,
        tke=tke_new,
        avm=Km,
        avt=Kt,
        mxlm=mxlm,
        mxld=mxld,
        pblh=pblh,
    )
