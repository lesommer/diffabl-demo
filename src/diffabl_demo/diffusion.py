"""Implicit vertical diffusion via Thomas algorithm.

Solves the tridiagonal system arising from Euler-backward discretization
of the 1D vertical diffusion equation on a staggered grid.
"""

import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
from diffabl_demo.grid import ABLGrid


def _build_diffusion_tridiag(
    K: jnp.ndarray,
    grid: ABLGrid,
    dt: float,
    surface_coeff: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    K = K.at[0].set(0.0).at[-1].set(0.0)
    n = grid.jpka

    alpha = dt * jnp.concatenate([jnp.array([0.0]), K[:-1]]) / jnp.concatenate([jnp.array([1.0]), grid.e3w[:-1]])
    gamma = dt * jnp.concatenate([K[1:], jnp.array([0.0])]) / jnp.concatenate([grid.e3w[1:], jnp.array([1.0])])

    sub = -alpha
    sup = -gamma
    diag = grid.e3t + alpha + gamma

    diag = diag.at[0].set(1.0)
    sub = sub.at[0].set(0.0)
    sup = sup.at[0].set(0.0)
    diag = diag.at[1].add(surface_coeff)

    return sub, diag, sup


def _thomas_solve(
    sub: jnp.ndarray,
    diag: jnp.ndarray,
    sup: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    n = len(diag)

    def forward_step(carry, i):
        c_prev, d_prev = carry
        denom = diag[i] - sub[i] * c_prev
        c_new = jnp.where(i < n - 1, sup[i] / denom, 0.0)
        d_new = (rhs[i] - sub[i] * d_prev) / denom
        return (c_new, d_new), (c_new, d_new)

    c0 = sup[0] / diag[0]
    d0 = rhs[0] / diag[0]
    indices = jnp.arange(1, n)
    (_, _), (c_arr, d_arr) = lax.scan(forward_step, (c0, d0), indices)
    c = jnp.concatenate([jnp.array([c0]), c_arr])
    d = jnp.concatenate([jnp.array([d0]), d_arr])

    def backward_step(x_next, i):
        x_i = d[i] - c[i] * x_next
        return x_i, x_i

    x_last = d[-1]
    indices_back = jnp.arange(n - 2, -1, -1)
    _, x_arr = lax.scan(backward_step, x_last, indices_back)
    x = jnp.concatenate([x_arr[::-1], jnp.array([x_last])])
    return x


def diffuse_implicit(
    phi: jnp.ndarray,
    K: jnp.ndarray,
    grid: ABLGrid,
    dt: float,
    surface_coeff: float = 0.0,
    surface_rhs: float = 0.0,
) -> jnp.ndarray:
    rhs = grid.e3t * phi
    rhs = rhs.at[1].add(surface_rhs)
    sub, diag, sup = _build_diffusion_tridiag(K, grid, dt, surface_coeff)
    return _thomas_solve(sub, diag, sup, rhs)
