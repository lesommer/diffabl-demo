"""TKE turbulence closure for the ABL model.

Implements the 1.5-order TKE closure from Lemarie et al. (2021):
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


def _compute_shear_mag(u: jnp.ndarray, v: jnp.ndarray,
                       grid: ABLGrid) -> jnp.ndarray:
    du = jnp.diff(u)
    dv = jnp.diff(v)
    dz = grid.e3w[1:]
    S_mag = jnp.sqrt((du / dz) ** 2 + (dv / dz) ** 2)
    return jnp.concatenate([jnp.array([S_mag[0]]), S_mag])


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


def _bl89_search_up(tke: jnp.ndarray, N2_safe: jnp.ndarray,
                    shear_mag: jnp.ndarray, e_sqrt: jnp.ndarray,
                    ghw: jnp.ndarray, e3t: jnp.ndarray,
                    n: int, Rod: float) -> jnp.ndarray:
    def search_from_k(k):
        z_k = ghw[k]
        cf_init = -tke[k]
        l_up_init = ghw[-1] - ghw[k]
        found_init = jnp.bool_(False)

        def scan_fn(carry, j):
            cf_prev, l_up, found = carry
            active = (j > k) & (j < n - 1)

            buoy = 0.5 * e3t[j] * (
                N2_safe[j] * (ghw[j] - z_k)
                + N2_safe[j - 1] * (ghw[j - 1] - z_k))
            shear_term = 0.5 * e3t[j] * Rod * (
                e_sqrt[j] * shear_mag[j]
                + e_sqrt[j - 1] * shear_mag[j - 1])
            cf_curr = jnp.where(active, cf_prev + buoy + shear_term, cf_prev)

            crossed = (cf_curr >= 0) & active & ~found
            z1 = ghw[j - 1] - z_k
            z2 = ghw[j] - z_k
            l_interp = (z1 * cf_curr - z2 * cf_prev) / (cf_curr - cf_prev + 1e-30)
            l_up_new = jnp.where(crossed, jnp.maximum(l_interp, 1e-10), l_up)
            found_new = found | crossed

            return (cf_curr, l_up_new, found_new), None

        js = jnp.arange(1, n)
        (_, l_up_final, _), _ = lax.scan(
            scan_fn, (cf_init, l_up_init, found_init), js)
        return l_up_final

    return jax.vmap(search_from_k)(jnp.arange(n))


def _bl89_search_down(tke: jnp.ndarray, N2_safe: jnp.ndarray,
                      shear_mag: jnp.ndarray, e_sqrt: jnp.ndarray,
                      ghw: jnp.ndarray, e3t: jnp.ndarray,
                      n: int, Rod: float) -> jnp.ndarray:
    def search_from_k(k):
        z_k = ghw[k]
        cf_init = -tke[k]
        l_down_init = ghw[k] - ghw[1]
        found_init = jnp.bool_(False)

        def scan_fn(carry, j):
            cf_prev, l_down, found = carry
            active = (j < k) & (j >= 1)

            buoy = 0.5 * e3t[j + 1] * (
                N2_safe[j + 1] * (z_k - ghw[j + 1])
                + N2_safe[j] * (z_k - ghw[j]))
            shear_term = 0.5 * e3t[j + 1] * Rod * (
                e_sqrt[j + 1] * shear_mag[j + 1]
                + e_sqrt[j] * shear_mag[j])
            cf_curr = jnp.where(active, cf_prev + buoy + shear_term, cf_prev)

            crossed = (cf_curr >= 0) & active & ~found
            z1 = z_k - ghw[j + 1]
            z2 = z_k - ghw[j]
            l_interp = (z1 * cf_prev - z2 * cf_curr) / (cf_prev - cf_curr + 1e-30)
            l_down_new = jnp.where(crossed, jnp.maximum(l_interp, 1e-10), l_down)
            found_new = found | crossed

            return (cf_curr, l_down_new, found_new), None

        js = jnp.arange(n - 2, 0, -1)
        (_, l_down_final, _), _ = lax.scan(
            scan_fn, (cf_init, l_down_init, found_init), js)
        return l_down_final

    return jax.vmap(search_from_k)(jnp.arange(n))


def _mixing_length_bl89(tke: jnp.ndarray, theta_v: jnp.ndarray,
                        N2: jnp.ndarray, shear_mag: jnp.ndarray,
                        ghw: jnp.ndarray, e3t: jnp.ndarray, e3w: jnp.ndarray,
                        jpka: int, mxl_min: float, rn_Lsfc: float,
                        Rod: float = 0.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = jpka
    N2_safe = jnp.maximum(N2, 1e-10)
    e_sqrt = jnp.sqrt(jnp.maximum(tke, 1e-10))

    l_up = _bl89_search_up(tke, N2_safe, shear_mag, e_sqrt, ghw, e3t, n, Rod)
    l_down = _bl89_search_down(tke, N2_safe, shear_mag, e_sqrt, ghw, e3t, n, Rod)

    l_up = l_up.at[0].set(ghw[1] * rn_Lsfc)
    l_down = l_down.at[0].set(ghw[1] * rn_Lsfc)
    l_up = l_up.at[-1].set(mxl_min)
    l_down = l_down.at[-1].set(mxl_min)

    l_up = jnp.maximum(l_up, 1e-10)
    l_down = jnp.maximum(l_down, 1e-10)

    l_m = 2.0 * jnp.sqrt(2.0) * (l_down ** (-2.0 / 3.0) + l_up ** (-2.0 / 3.0)) ** (-1.5)
    l_d = jnp.minimum(l_down, l_up)

    l_m = jnp.maximum(l_m, mxl_min)
    l_d = jnp.maximum(l_d, mxl_min)

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

    if params.nn_amxl <= 1:
        l_m, l_d = _mixing_length_deardorff(tke, N2, grid.ghw, grid.e3t, n)
    else:
        shear_mag = _compute_shear_mag(u, v, grid)
        Rod = params.Rod if params.nn_amxl == 3 else 0.0
        l_m, l_d = _mixing_length_bl89(
            tke, theta_v, N2, shear_mag,
            grid.ghw, grid.e3t, grid.e3w, n,
            params.mxl_min, params.rn_Lsfc, Rod)

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
