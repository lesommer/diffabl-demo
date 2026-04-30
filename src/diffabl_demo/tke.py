"""TKE turbulence closure for the ABL model.

Implements the 1.5-order TKE closure from Lemarié et al. (2021):
  - TKE prognostic equation with Patankar positivity treatment
  - Four mixing-length options (Deardorff, Modified Deardorff, BL89, Modified BL89)
  - Stability function and eddy viscosity/diffusivity diagnostics
  - PBL height diagnosis via bulk Richardson number
"""

import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
from diffabl_demo.grid import ABLGrid
from diffabl_demo.state import ABLParams
from diffabl_demo.diffusion import _thomas_solve


def _compute_shear(u: jnp.ndarray, v: jnp.ndarray,
                   grid: ABLGrid) -> jnp.ndarray:
    du = jnp.diff(u)
    dv = jnp.diff(v)
    dz = grid.e3w[1:]
    S2 = (du / dz) ** 2 + (dv / dz) ** 2
    return jnp.concatenate([jnp.array([S2[0]]), S2])


def _compute_N2(theta: jnp.ndarray, q: jnp.ndarray,
                grid: ABLGrid, params: ABLParams) -> jnp.ndarray:
    theta_v = theta * (1.0 + 0.61 * q)
    dtheta_v = jnp.diff(theta_v)
    dz = grid.e3w[1:]
    N2 = params.g / theta_v[1:] * (dtheta_v / dz + 0.61 * theta[1:] * jnp.diff(q) / dz)
    return jnp.concatenate([jnp.array([N2[0]]), N2])


def _diagnose_pblh(S2: jnp.ndarray, N2: jnp.ndarray, f: float,
                   ghw: jnp.ndarray, e3w: jnp.ndarray,
                   params: ABLParams, pblh_prev: jnp.ndarray) -> jnp.ndarray:
    eps_sfc = params.eps_sfc
    sigma = ghw / jnp.maximum(pblh_prev, 1.0)
    weight = sigma / (sigma + eps_sfc)
    integrand = S2 - N2 / params.Ric - params.Cek * f ** 2
    FC = jnp.cumsum(integrand * weight * e3w)

    neg = FC < 0
    idx_first = jnp.argmax(neg)
    found = jnp.any(neg)

    k = idx_first
    k_prev = jnp.maximum(k - 1, 0)
    z_prev = ghw[k_prev]
    z_curr = ghw[k]
    fc_prev = FC[k_prev]
    fc_curr = FC[k]
    dz_interp = (z_curr - z_prev) * fc_prev / (fc_prev - fc_curr + 1e-30)
    h = z_prev + dz_interp
    return jnp.where(found, jnp.clip(h, ghw[1], ghw[-1]), ghw[-1])


def _mixing_length_deardorff(tke: jnp.ndarray, N2: jnp.ndarray,
                              ghw: jnp.ndarray, e3t: jnp.ndarray,
                              jpka: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    N2_safe = jnp.maximum(N2, 1e-10)
    l_diag = jnp.sqrt(2.0 * jnp.maximum(tke, 1e-10) / N2_safe)
    l_diag = jnp.minimum(l_diag, ghw)
    l_diag = jnp.minimum(l_diag, ghw[-1] - ghw)

    l_up = l_diag.copy()
    l_down = l_diag.copy()

    def sweep_up(l_prev, k):
        l_new = jnp.minimum(l_diag[k], l_prev + e3t[k])
        return l_new, l_new

    _, l_up_arr = lax.scan(sweep_up, l_diag[0], jnp.arange(1, jpka))
    l_up = l_up.at[1:].set(l_up_arr)

    def sweep_down(l_prev, k):
        l_new = jnp.minimum(l_diag[k], l_prev + e3t[k])
        return l_new, l_new

    _, l_down_arr = lax.scan(sweep_down, l_diag[-1], jnp.arange(jpka - 2, -1, -1))
    l_down = l_down.at[:-1].set(l_down_arr)

    l_up_p = jnp.maximum(l_up, 1e-10)
    l_down_p = jnp.maximum(l_down, 1e-10)
    l_m = 2.0 * jnp.sqrt(2.0) * (l_down_p ** (-2.0 / 3.0) + l_up_p ** (-2.0 / 3.0)) ** (-1.5)
    l_d = jnp.minimum(l_down, l_up)
    return l_m, l_d


def _mixing_length_bl89(tke: jnp.ndarray, theta_v: jnp.ndarray,
                         ghw: jnp.ndarray, e3t: jnp.ndarray, e3w: jnp.ndarray,
                         jpka: int, Rod: float = 0.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    beta = 9.81 / jnp.mean(theta_v)

    def search_up(k):
        e_k = tke[k]
        integral = 0.0
        dz_acc = 0.0
        for j in range(k, jpka - 1):
            dtheta = theta_v[j + 1] - theta_v[j] if j + 1 < len(theta_v) else 0.0
            dz = e3t[j + 1] if j + 1 < len(e3t) else 0.0
            buoy = beta * dtheta
            shear = Rod * jnp.sqrt(jnp.maximum(e_k, 0.0)) * 0.0
            consumption = (buoy - shear)
            integral_prev = integral
            integral = integral + consumption * dz
            dz_acc = dz_acc + dz
            crossed = integral >= e_k
            frac = jnp.where(crossed & (integral > integral_prev),
                             (e_k - integral_prev) / (integral - integral_prev + 1e-30), 0.0)
            l_up_k = jnp.where(crossed, dz_acc - dz + frac * dz, dz_acc)
        return l_up_k

    def search_down(k):
        e_k = tke[k]
        integral = 0.0
        dz_acc = 0.0
        for j in range(k, 0, -1):
            dtheta = theta_v[j] - theta_v[j - 1] if j > 0 else 0.0
            dz = e3t[j] if j < len(e3t) else 0.0
            buoy = beta * dtheta
            shear = Rod * jnp.sqrt(jnp.maximum(e_k, 0.0)) * 0.0
            consumption = (buoy - shear)
            integral_prev = integral
            integral = integral + consumption * dz
            dz_acc = dz_acc + dz
            crossed = integral >= e_k
            frac = jnp.where(crossed & (integral > integral_prev),
                             (e_k - integral_prev) / (integral - integral_prev + 1e-30), 0.0)
            l_down_k = jnp.where(crossed, dz_acc - dz + frac * dz, dz_acc)
        return l_down_k

    l_up = jax.vmap(search_up)(jnp.arange(jpka))
    l_down = jax.vmap(search_down)(jnp.arange(jpka))

    l_up = jnp.maximum(l_up, 1e-10)
    l_down = jnp.maximum(l_down, 1e-10)
    l_m = 2.0 * jnp.sqrt(2.0) * (l_down ** (-2.0 / 3.0) + l_up ** (-2.0 / 3.0)) ** (-1.5)
    l_d = jnp.minimum(l_down, l_up)
    return l_m, l_d


def abl_zdf_tke(state, grid: ABLGrid, params: ABLParams,
                ustar2: float) -> tuple:
    u = state.u
    v = state.v
    theta = state.theta
    q = state.q
    tke = state.tke
    avm_prev = state.avm

    S2 = _compute_shear(u, v, grid)
    N2 = _compute_N2(theta, q, grid, params)
    theta_v = theta * (1.0 + 0.61 * q)

    n = grid.jpka
    dt = params.dt

    l_m, l_d = _mixing_length_deardorff(tke, N2, grid.ghw, grid.e3t, n)

    e_sqrt = jnp.sqrt(jnp.maximum(tke, params.tke_min))
    Km_diag = jnp.maximum(params.Cm * l_m * e_sqrt, params.avm_bak)
    Kt_diag = jnp.maximum(params.Ct * l_m * e_sqrt, params.avt_bak)

    Km_diff = 0.5 * (Km_diag[:-1] + Km_diag[1:])
    Km_diff = jnp.concatenate([jnp.array([Km_diag[0]]), Km_diff])
    Kt_diff = 0.5 * (Kt_diag[:-1] + Kt_diag[1:])
    Kt_diff = jnp.concatenate([jnp.array([Kt_diag[0]]), Kt_diff])

    shear_prod = Km_diag * S2
    buoy_prod = -Kt_diag * N2

    Km_w = 0.5 * (Km_diag[:-1] + Km_diag[1:])
    Kt_w = 0.5 * (Kt_diag[:-1] + Kt_diag[1:])

    K_tke = params.Ce * l_m * e_sqrt
    K_tke = K_tke.at[0].set(0.0).at[-1].set(0.0)

    alpha = dt * jnp.concatenate([jnp.array([0.0]), K_tke[:-1]]) / jnp.concatenate([jnp.array([1.0]), grid.e3w[:-1]])
    gamma = dt * jnp.concatenate([K_tke[1:], jnp.array([0.0])]) / jnp.concatenate([grid.e3w[1:], jnp.array([1.0])])

    sub = -alpha
    sup = -gamma
    diag = grid.e3t + alpha + gamma
    rhs = grid.e3t * tke

    e_sfc = ustar2 / jnp.sqrt(params.Cm * params.Ceps)
    diag = diag.at[0].set(1.0)
    sub = sub.at[0].set(0.0)
    sup = sup.at[0].set(0.0)
    rhs = rhs.at[0].set(e_sfc)

    net_destroy = shear_prod + buoy_prod < 0
    patankar = jnp.where(net_destroy, jnp.abs(buoy_prod) / jnp.maximum(tke, params.tke_min), 0.0)
    patankar = patankar.at[0].set(0.0)
    diag = diag + dt * patankar

    dissipation = params.Ceps * e_sqrt / jnp.maximum(l_m, 1e-10)
    dissipation = dissipation.at[0].set(0.0)
    diag = diag + dt * dissipation

    rhs = rhs + dt * shear_prod.at[0].set(0.0)
    rhs = rhs + dt * jnp.where(net_destroy, 0.0, buoy_prod).at[0].set(0.0)

    tke_new = _thomas_solve(sub, diag, sup, rhs)
    tke_new = jnp.maximum(tke_new, params.tke_min)

    pblh = _diagnose_pblh(S2, N2, params.f, grid.ghw, grid.e3w, params, state.pblh[0])

    phi_z = 1.0 / (1.0 + jnp.maximum(params.phi_max, params.Ric * l_m * l_d * N2 / jnp.maximum(tke_new, params.tke_min)))
    Km = jnp.maximum(params.Cm * l_m * jnp.sqrt(jnp.maximum(tke_new, params.tke_min)), params.avm_bak)
    Kt = jnp.maximum(params.Ct * l_m * jnp.sqrt(jnp.maximum(tke_new, params.tke_min)) * phi_z, params.avt_bak)

    return tke_new, Km, Kt, l_m, l_d, jnp.array([pblh])
