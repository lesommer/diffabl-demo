"""Height-dependent Newtonian nudging toward large-scale reference."""

import jax.numpy as jnp
from diffabl_demo.grid import ABLGrid


def nudging_coefficient(
    ght: jnp.ndarray,
    pblh: float,
    alpha_min: float,
    alpha_max: float,
) -> jnp.ndarray:
    sigma = ght / jnp.maximum(pblh, 1.0)
    return alpha_min + (alpha_max - alpha_min) * sigma ** 3


def apply_nudging(
    phi: jnp.ndarray,
    phi_ref: jnp.ndarray,
    alpha: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    factor = 1.0 / (1.0 + dt * alpha)
    return factor * (phi + dt * alpha * phi_ref)
