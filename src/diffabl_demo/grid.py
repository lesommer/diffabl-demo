"""Vertical grid construction for the ABL model.

Convention (matching NEMO ABL):
  - Level 0 (jk=1 in Fortran): surface boundary, ght=0, ghw=0, e3t=0
  - Levels 1..K-1: interior levels carrying u, v, theta, q at t-points
  - ghw[-1] = domain top (zhmax for sinh, height for uniform)
"""

import jax.numpy as jnp
import equinox as eqx


class ABLGrid(eqx.Module):
    ght: jnp.ndarray
    ghw: jnp.ndarray
    e3t: jnp.ndarray
    e3w: jnp.ndarray
    jpka: int
    jpkam1: int

    @property
    def height(self) -> float:
        return float(self.ghw[-1])

    @property
    def n_levels(self) -> int:
        return self.jpka


def _build_grid_from_ghw(ghw: jnp.ndarray, jpka: int) -> "ABLGrid":
    ght = jnp.concatenate([
        jnp.array([0.0]),
        0.5 * (ghw[:-1] + ghw[1:]),
    ])
    e3t = jnp.concatenate([
        jnp.array([0.0]),
        jnp.diff(ghw),
    ])
    e3w = jnp.concatenate([
        jnp.diff(ght),
        jnp.array([ghw[-1] - ght[-1]]),
    ])
    return ABLGrid(ght=ght, ghw=ghw, e3t=e3t, e3w=e3w,
                   jpka=jpka, jpkam1=jpka - 1)


def uniform_grid(n_levels: int, height: float) -> ABLGrid:
    dz = height / (n_levels - 1)
    ghw = dz * jnp.arange(n_levels)
    return _build_grid_from_ghw(ghw, n_levels)


def sinh_stretched_grid(n_levels: int, zhmax: float, zhc: float,
                        theta_s: float) -> ABLGrid:
    zds = 1.0 / n_levels
    zcff = (zhmax - zhc) / jnp.sinh(theta_s)

    jk = jnp.arange(n_levels - 1, 0, -1)
    zsc_w = zds * jk
    ghw_upper = zhc * zsc_w + zcff * jnp.sinh(theta_s * zsc_w)

    ghw = jnp.concatenate([jnp.array([0.0]), ghw_upper[::-1]])
    return _build_grid_from_ghw(ghw, n_levels)
