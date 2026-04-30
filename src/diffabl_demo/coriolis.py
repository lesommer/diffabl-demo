"""Coriolis treatment for the ABL model.

Implements:
  - Forward-backward scheme (alternating order on even/odd steps)
  - Semi-implicit scheme with geostrophic guide
"""

import jax.numpy as jnp
from diffabl_demo.state import ABLParams


def coriolis_forward_backward(
    u: jnp.ndarray,
    v: jnp.ndarray,
    f: float,
    dt: float,
    step: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    fdt = f * dt
    even = step % 2 == 0
    if even:
        u_new = u + fdt * v
        v_new = v - fdt * u_new
    else:
        v_new = v - fdt * u
        u_new = u + fdt * v_new
    return u_new, v_new


def coriolis_semi_implicit(
    u: jnp.ndarray,
    v: jnp.ndarray,
    f: float,
    dt: float,
    u_geo: float = 0.0,
    v_geo: float = 0.0,
    gamma: float = 0.55,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    fdt2 = (f * dt) ** 2
    g = gamma
    denom_u = 1.0 + g ** 2 * fdt2
    u_new = ((1.0 - g * (1.0 - g) * fdt2) * u +
             f * dt * (v - v_geo) +
             g * fdt2 * u_geo) / denom_u
    v_new = ((1.0 - g * (1.0 - g) * fdt2) * v -
             f * dt * (u - u_geo) +
             g * fdt2 * v_geo) / denom_u
    return u_new, v_new


def apply_coriolis(
    u: jnp.ndarray,
    v: jnp.ndarray,
    params: ABLParams,
    step: int,
    u_geo: float = 0.0,
    v_geo: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if params.SemiImp_Cor:
        return coriolis_semi_implicit(u, v, params.f, params.dt,
                                      u_geo, v_geo, params.gamma_cor)
    return coriolis_forward_backward(u, v, params.f, params.dt, step)
