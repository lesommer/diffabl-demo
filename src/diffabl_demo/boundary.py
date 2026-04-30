"""Surface boundary conditions and bulk formulae for the ABL model.

Implements simplified bulk aerodynamic formulae for momentum, heat,
and moisture fluxes at the air-sea interface. All functions are
JAX-compatible (no Python float/int conversions on traced values).
"""

import jax.numpy as jnp
from diffabl_demo.state import ABLParams


def bulk_Cd(wind_speed: jnp.ndarray, params: ABLParams | None = None) -> jnp.ndarray:
    U = jnp.maximum(wind_speed, 0.5)
    return (0.75 + 0.067 * U) * 1e-3


def bulk_Ch(wind_speed: jnp.ndarray, params: ABLParams | None = None) -> jnp.ndarray:
    return bulk_Cd(wind_speed, params)


def bulk_Ce(wind_speed: jnp.ndarray, params: ABLParams | None = None) -> jnp.ndarray:
    return bulk_Cd(wind_speed, params)


def compute_ustar2(u: jnp.ndarray, v: jnp.ndarray,
                   u_ocean: float, v_ocean: float,
                   wind_speed: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Cd = bulk_Cd(wind_speed)
    du = u - u_ocean
    dv = v - v_ocean
    ustar2 = Cd * (du ** 2 + dv ** 2)
    return ustar2, Cd, wind_speed


def surface_momentum_coeff(Cd: jnp.ndarray, wind_speed: jnp.ndarray, dt: float) -> jnp.ndarray:
    return dt * Cd * wind_speed


def surface_heat_coeff(Ch: jnp.ndarray, wind_speed: jnp.ndarray, dt: float) -> jnp.ndarray:
    return dt * Ch * wind_speed


def surface_moisture_coeff(Ce: jnp.ndarray, wind_speed: jnp.ndarray, dt: float) -> jnp.ndarray:
    return dt * Ce * wind_speed
