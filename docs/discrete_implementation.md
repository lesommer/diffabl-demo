# Discrete Implementation — Simplified ABL Model

Reference: Lemarié et al. (2021), Geosci. Model Dev., 14, 543–572.
FORTRAN reference: `nemo_abl1d_GMD_2020/NEMO_CODE/src/ABL/ablmod.F90`

## 1. Vertical Grid

The model uses a staggered 1D vertical grid with $K$ levels (indexed $k = 1, \ldots, K$):

- **T-levels** (cell centers): $z^t_k = \text{ght}(k)$ — carry $u, v, \theta, q$
- **W-levels** (cell interfaces): $z^w_k = \text{ghw}(k)$ — carry $e, K_m, K_t, l_m, l_d$

Grid spacing: $\Delta z^t_k = \text{e3t}(k)$, $\Delta z^w_k = \text{e3w}(k)$.

Two grid types:

### 1.1 Uniform Grid
$$z^t_k = (k - 0.5) \cdot \Delta z, \quad \Delta z = H / K$$

### 1.2 Sinh-Stretched Grid
$$z^t_k = z_{\max} \frac{\sinh(\theta_s\, \hat{z}_k / z_c)}{\sinh(\theta_s)}$$

where $\hat{z}_k \in [0, z_c]$ is a normalized coordinate, $z_c$ is the stretching center, $\theta_s$ is the stretching factor, and $z_{\max}$ is the domain top.

## 2. Time Stepping

Euler-backward (first-order implicit) for all diffusion terms. The overall sequence per time step $\Delta t$ is:

```
1. Compute u*, roughness length from surface drag
2. TKE step: abl_zdf_tke()
   a. Compute shear S^2 at w-levels
   b. Compute N^2 at w-levels
   c. Build and solve TKE tridiagonal (with Patankar trick)
   d. Diagnose PBL height h
   e. Diagnose mixing length l_m, l_d
   f. Compute Km, Kt
3. Tracer step: implicit diffusion for θ and q
4. Coriolis step (forward-backward or semi-implicit)
5. Momentum step: implicit diffusion for u and v
6. Nudging step: relaxation toward large-scale reference
7. Swap time indices (nt_n ↔ nt_a)
```

## 3. Discrete Operators

### 3.1 Vertical Derivatives (w-levels from t-levels)

$$\left.\frac{\partial \phi}{\partial z}\right|_{z^w_k} \approx \frac{\phi_{k} - \phi_{k-1}}{z^t_k - z^t_{k-1}}$$

for $k = 2, \ldots, K$ (w-levels between t-levels).

### 3.2 Vertical Derivatives (t-levels from w-levels)

$$\left.\frac{\partial \psi}{\partial z}\right|_{z^t_k} \approx \frac{\psi_{k+1} - \psi_k}{z^w_{k+1} - z^w_k}$$

### 3.3 Vertical Diffusion Operator (implicit, at t-level k)

$$\frac{\partial}{\partial z}\!\left(K\,\frac{\partial \phi}{\partial z}\right)\bigg|_{z^t_k} \approx \frac{1}{\Delta z^t_k}\left[\frac{K_{k}\,(\phi_{k+1} - \phi_k)}{\Delta z^w_k} - \frac{K_{k-1}\,(\phi_k - \phi_{k-1})}{\Delta z^w_{k-1}}\right]$$

where $K_k$ and $K_{k-1}$ are diffusion coefficients at the w-levels above and below t-level $k$.

## 4. Implicit Tridiagonal System

For a generic variable $\phi$ with diffusion coefficient $K$ on w-levels, the implicit Euler-backward discretization yields:

$$-\alpha_{k-1}\,\phi^{n+1}_{k-1} + \beta_k\,\phi^{n+1}_k - \gamma_k\,\phi^{n+1}_{k+1} = \Delta z^t_k\,\phi^n_k$$

where:

$$\alpha_{k-1} = \frac{\Delta t\, K_{k-1}}{\Delta z^w_{k-1}}, \quad \gamma_k = \frac{\Delta t\, K_k}{\Delta z^w_k}$$

$$\beta_k = \Delta z^t_k + \alpha_{k-1} + \gamma_k + \text{surface term (if } k = 1\text{)}$$

### 4.1 Surface BC (k = 1)

For tracers with Robin BC (e.g., temperature):
$$\beta_1 = \Delta z^t_1 + \gamma_1 + \Delta t\, C_h\, |\mathbf{U}|$$
$$\text{RHS}_1 = \Delta z^t_1\, \theta^n_1 + \Delta t\, C_h\, |\mathbf{U}|\, \theta_{\text{sfc}}$$

For momentum with drag BC:
$$\beta_1 = \Delta z^t_1 + \gamma_1 + \Delta t\, C_d\, |\mathbf{U}|$$
$$\text{RHS}_1 = \Delta z^t_1\, u^n_1 + \Delta t\, C_d\, |\mathbf{U}|\, u_{\text{ocean}}$$

### 4.2 Top BC (k = K)

Neumann (zero flux): set $K_K = 0$ in the coefficients, so $\gamma_K = 0$.

### 4.3 Thomas Algorithm

Forward sweep (k = 1 to K):
$$\tilde{\beta}_1 = \beta_1, \quad \tilde{d}_1 = \text{RHS}_1$$
$$\tilde{\beta}_k = \beta_k - \frac{\alpha_{k-1}\,\gamma_{k-1}}{\tilde{\beta}_{k-1}}, \quad \tilde{d}_k = d_k - \frac{\alpha_{k-1}\,\tilde{d}_{k-1}}{\tilde{\beta}_{k-1}}$$

Back substitution (k = K to 1):
$$\phi^{n+1}_K = \frac{\tilde{d}_K}{\tilde{\beta}_K}$$
$$\phi^{n+1}_k = \frac{\tilde{d}_k - \gamma_k\,\phi^{n+1}_{k+1}}{\tilde{\beta}_k}$$

## 5. TKE Equation Discretization

### 5.1 Shear Production

At w-level $k$ (between t-levels $k$ and $k-1$):

$$P_s(k) = K_m(k) \left[\left(\frac{u_k - u_{k-1}}{\Delta z^w_k}\right)^2 + \left(\frac{v_k - v_{k-1}}{\Delta z^w_k}\right)^2\right]$$

Note: the shear is squared *after* dividing by grid spacing.

### 5.2 Buoyancy Production

$$P_b(k) = -K_t(k)\, N^2(k)$$

where $N^2(k)$ is computed at w-level $k$ from virtual potential temperature gradient across adjacent t-levels.

### 5.3 Patankar Treatment

When $P_s(k) + P_b(k) < 0$ (net destruction), the buoyancy term is moved to the LHS of the tridiagonal to be solved implicitly:

$$\text{LHS diagonal contribution:} \quad +\frac{\Delta t\, |P_b(k)|}{e_k}$$

This prevents overshoot to negative TKE values without clipping.

### 5.4 Dissipation

$$\varepsilon(k) = C_\varepsilon\, \frac{\sqrt{e_k}}{l_m(k)}\, e_k$$

Treated implicitly via Patankar: moved to LHS diagonal as:

$$+\frac{\Delta t\, C_\varepsilon\, \sqrt{e_k}}{l_m(k)}$$

### 5.5 TKE Surface BC

$$e_1 = \frac{u_*^2}{\sqrt{C_m\, C_\varepsilon}}$$

imposed as a Dirichlet condition at the first w-level.

### 5.6 TKE Top BC

Neumann (zero flux): no TKE diffusion through the domain top.

## 6. Mixing Length Discretization

### 6.0 Deardorff — Sweep Algorithm

For each w-level $k$ (ascending from surface):

**Diagnostic length:**
$$l_{\text{diag}}(k) = \min\!\left(\sqrt{\frac{2\,e_k}{N^2(k) + \epsilon_N}},\; z^w_k,\; z^w_K - z^w_k\right)$$

**Upward sweep** (k = 1 to K):
$$l_{\text{up}}(k) = \min\!\left(l_{\text{diag}}(k),\; l_{\text{up}}(k-1) + \Delta z^t_k\right)$$

**Downward sweep** (k = K to 1):
$$l_{\text{down}}(k) = \min\!\left(l_{\text{diag}}(k),\; l_{\text{down}}(k+1) + \Delta z^t_{k+1}\right)$$

**Final:**
$$l_m(k) = 2\sqrt{2}\left[l_{\text{down}}(k)^{-2/3} + l_{\text{up}}(k)^{-2/3}\right]^{-3/2}$$
$$l_d(k) = \min\!\left(l_{\text{down}}(k), l_{\text{up}}(k)\right)$$

### 6.2 BL89 — Search Algorithm

For each w-level $k$, search upward from $k$ to find the height $z_{\text{up}}$ where the accumulated buoyancy consumption equals the available TKE:

$$\sum_{j=k}^{j^*} \beta\, \frac{\theta_{v,j+1} - \theta_{v,j}}{\Delta z}\, \Delta z = e_k$$

with linear interpolation between levels $j^*$ and $j^*+1$ when the zero-crossing is between grid points. Then:

$$l_{\text{up}}(k) = z^w_{j^*} - z^w_k + \text{fractional level correction}$$

Analogous downward search for $l_{\text{down}}(k)$.

## 7. PBL Height Diagnosis

Compute the cumulative function at each w-level:

$$\text{FC}(k) = \sum_{j=1}^{k} \left[S^2(j) - \frac{1}{R_{ic}} N^2(j) - C_{ek}\, f^2\right] w\!\left(\frac{z^w_j}{h^{n}}\right) \Delta z^w_j$$

where $h^n$ is the PBL height from the previous time step and the weighting function is:

$$w(\zeta) = \frac{\zeta}{\zeta + \varepsilon_{\text{sfc}}}, \quad \varepsilon_{\text{sfc}} = \frac{1}{1 + 2.8^2}$$

Find $k^*$ where FC first becomes negative, then interpolate:

$$h^{n+1} = z^w_{k^*-1} + \frac{\text{FC}(k^*-1)}{\text{FC}(k^*-1) - \text{FC}(k^*)}\, (z^w_{k^*} - z^w_{k^*-1})$$

On the first time step, use $h^0 = 1000$ m (default) as initial guess.

## 8. Coriolis Discretization

### 8.1 Forward-Backward (default)

On even time steps ($n$ even):
```
u* = u^n + Δt · f · v^n          (explicit u update)
v^{n+1} = v^n - Δt · f · u*      (v uses updated u*)
```

On odd time steps ($n$ odd):
```
v* = v^n - Δt · f · u^n
u^{n+1} = u^n + Δt · f · v*
```

### 8.2 Semi-Implicit with Geostrophic Guide

$$u^{n+1} = \frac{(1 - \gamma(1-\gamma)(f\Delta t)^2)\, u^n + f\Delta t\,(v^n - v_g) + \gamma\,(f\Delta t)^2\, u_g}{1 + \gamma^2 (f\Delta t)^2}$$

$$v^{n+1} = \frac{(1 - \gamma(1-\gamma)(f\Delta t)^2)\, v^n - f\Delta t\,(u^n - u_g) + \gamma\,(f\Delta t)^2\, v_g}{1 + \gamma^2 (f\Delta t)^2}$$

with $\gamma = 0.55$.

## 9. Nudging Discretization

At each t-level $k$:

$$\phi^{n+1}_k \leftarrow \phi^{n+1}_k - \Delta t\, \alpha_k\, (\phi^{n+1}_k - \phi_{\text{ref},k})$$

where:

$$\alpha_k = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min})\, \sigma_k^3, \quad \sigma_k = \frac{z^t_k}{h_{\text{pbl}}}$$

Applied after the diffusion step, as an implicit correction (absorbed into the diagonal for stability).

## 10. Vectorization Strategy (JAX)

- All vertical operations are naturally 1D per column
- When running multiple columns (2D domain), use `jax.vmap` over the horizontal dimensions
- The Thomas solver is applied column-wise; `lineax` tridiagonal solvers may be used as an alternative
- Mixing-length sweeps are sequential in the vertical — use `jax.lax.scan` for differentiability
- PBL height search is also sequential — `jax.lax.scan` or `jax.lax.while_loop`

## 11. Differentiability Considerations

All operations must be compatible with JAX autodiff:

| Operation | Concern | Strategy |
|-----------|---------|----------|
| Thomas algorithm | Implicit solve | JAX differentiates through the solve automatically |
| Patankar trick | Conditional (if Ps+Pb < 0) | Use `jax.numpy.where` instead of Python `if` |
| Mixing length sweeps | Sequential | `jax.lax.scan` |
| PBL height search | Index-based interpolation | Use differentiable interpolation (soft argmin or continuous relaxation) |
| `max()` clamps | Non-smooth | `jax.numpy.maximum` is differentiable a.e. |
| Time-step swap | Index toggle | Use `(nt_n + 1) % 2` arithmetic instead of `if` |

The solver must be a pure function: `state_new = step(state, params, forcing)` with no side effects, enabling `jax.grad(step, argnums=...)` and `jax.jacfwd(step, ...)`.
