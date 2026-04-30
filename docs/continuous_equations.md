# Continuous Equations — Simplified ABL Model

Reference: Lemarié et al. (2021), Geosci. Model Dev., 14, 543–572.
Equation numbers below refer to this paper where applicable.

## 1. Governing Equations

The model solves a column-integrated (1D vertical) set of equations for horizontal momentum, potential temperature, specific humidity, and turbulent kinetic energy on a staggered vertical grid.

### 1.1 Momentum

$$
\frac{\partial u}{\partial t} = f v - \frac{\partial}{\partial z}\!\left(\overline{u'w'}\right) + F^u_{\text{geo}} + F^u_{\text{nudge}}
$$

$$
\frac{\partial v}{\partial t} = -f u - \frac{\partial}{\partial z}\!\left(\overline{v'w'}\right) + F^v_{\text{geo}} + F^v_{\text{nudge}}
$$

where:
- $f$ is the Coriolis parameter
- $\overline{u'w'} = -K_m \,\partial u/\partial z$ and $\overline{v'w'} = -K_m \,\partial v/\partial z$ are the Reynolds fluxes
- $F^u_{\text{geo}}$, $F^v_{\text{geo}}$ are the large-scale pressure gradient / geostrophic forcing terms
- $F^u_{\text{nudge}}$, $F^v_{\text{nudge}}$ are Newtonian relaxation toward large-scale winds

### 1.2 Thermodynamic Variables

$$
\frac{\partial \theta}{\partial t} = -\frac{\partial}{\partial z}\!\left(\overline{w'\theta'}\right) + F^{\theta}_{\text{nudge}}
$$

$$
\frac{\partial q}{\partial t} = -\frac{\partial}{\partial z}\!\left(\overline{w'q'}\right) + F^{q}_{\text{nudge}}
$$

where:
- $\overline{w'\theta'} = -K_t \,\partial\theta/\partial z$ and $\overline{w'q'} = -K_t \,\partial q/\partial z$
- $K_t$ is the turbulent diffusivity

### 1.3 Turbulent Kinetic Energy

$$
\frac{\partial e}{\partial t} = K_m \left[\left(\frac{\partial u}{\partial z}\right)^2 + \left(\frac{\partial v}{\partial z}\right)^2\right] - K_t N^2 - C_\varepsilon \frac{\sqrt{e}}{l_m}\, e + \frac{\partial}{\partial z}\!\left(C_e \, l_m \sqrt{e} \,\frac{\partial e}{\partial z}\right)
$$

where the four RHS terms are respectively:
1. **Shear production**: $P_s = K_m \left[(\partial u/\partial z)^2 + (\partial v/\partial z)^2\right]$
2. **Buoyancy production/destruction**: $P_b = -K_t N^2$
3. **Dissipation**: $\varepsilon = C_\varepsilon \sqrt{e}\, e / l_m$
4. **Vertical diffusion of TKE**: $D_e = \partial_z(C_e l_m \sqrt{e}\, \partial_z e)$

## 2. Buoyancy Frequency

The Brunt–Väisälä frequency with virtual potential temperature effects:

$$
N^2 = \frac{g}{\theta_v}\left(\frac{\partial \theta}{\partial z} + 0.61\,\theta\frac{\partial q}{\partial z}\right)
$$

where the virtual potential temperature is:

$$
\theta_v = \theta\,(1 + 0.61\, q)
$$

## 3. Turbulent Viscosity and Diffusivity

### 3.1 Stability Function

$$
\phi_z = \frac{1}{1 + \max\!\left(\phi_{\max},\; R_{ic}\, \frac{l_m\, l_d\, N^2}{e}\right)}
$$

where $\phi_{\max} = (1 - 2.2) / 2.2 \approx -0.545$ acts as a floor to prevent negative diffusivities.

### 3.2 Eddy Coefficients

$$
K_m = \max\!\left(C_m\, l_m\, \sqrt{e},\; K_m^{\text{bak}}\right)
$$

$$
K_t = \max\!\left(C_t\, l_m\, \sqrt{e}\, \phi_z,\; K_t^{\text{bak}}\right)
$$

## 4. Mixing Length Diagnostics

Four options are available (controlled by parameter `nn_amxl`):

### 4.0 Deardorff (1980)

Diagnostic length:
$$
l_{\text{diag}} = \sqrt{\frac{2\,e}{N^2 + \epsilon_N}}
$$

Limited by distance to surface ($z$) and domain top ($H - z$). Upward and downward sweeps enforce monotonicity with grid spacing. Final mixing length:

$$
l_m = 2\sqrt{2} \left[l_{\text{down}}^{-2/3} + l_{\text{up}}^{-2/3}\right]^{-3/2}
$$

Dissipative length: $l_d = \min(l_{\text{down}}, l_{\text{up}})$.

### 4.1 Modified Deardorff (Rodier et al.)

Diagnostic length:
$$
l_{\text{diag}} = \frac{2\sqrt{e}}{R_{od}\,|S| + \sqrt{R_{od}^2\,S^2 + 2\,N^2}}
$$

where $S = \sqrt{(\partial u/\partial z)^2 + (\partial v/\partial z)^2}$ is the wind shear and $R_{od}$ is the Rodier parameter. Same sweep procedure as Deardorff.

### 4.2 Bougeault & Lacarrère (1989) — BL89

Upward length $l_{\text{up}}(k)$: distance from level $k$ to the height where the TKE at $k$ would be consumed by buoyancy when searching upward:

$$
\int_z^{z+l_{\text{up}}} \beta\, \frac{\partial \theta_v}{\partial z}\, dz' = e(z)
$$

with linear interpolation when the zero-crossing is between grid points. Downward length $l_{\text{down}}$ computed analogously. Same harmonic combination formula.

### 4.3 Modified BL89 with Shear

Like BL89 but the search integral includes shear production:

$$
\int_z^{z+l} \left[\beta\, \frac{\partial \theta_v}{\partial z} - R_{od}\, \sqrt{e}\, |S|\right] dz' = e(z)
$$

## 5. PBL Height Diagnosis

The boundary layer height $h$ is diagnosed from a bulk Richardson number criterion:

$$
\text{FC}(z) = \int_0^z \left[S^2(z') - \frac{1}{R_{ic}} N^2(z') - C_{ek}\, f^2\right] w\!\left(\frac{z'}{h}\right) dz'
$$

where:
- $w(\zeta) = \zeta / (\zeta + \varepsilon_{\text{sfc}})$ is a surface-layer weighting with $\varepsilon_{\text{sfc}} = 1/(1 + 2.8^2)$
- $C_{ek} = 258$ is an Ekman correction constant
- $h$ is the PBL height from the previous time step (used in the weighting)

The PBL height is found where $\text{FC}$ first becomes negative, with linear interpolation between the adjacent grid levels.

## 6. Boundary Conditions

### 6.1 Surface (z = 0)

**Momentum** — Robin condition incorporating drag:
$$
\overline{u'w'}\big|_{z=0} = -C_d\, |\mathbf{U}|\, (u - u_{\text{ocean}})
$$

**Temperature** — Robin condition:
$$
\overline{w'\theta'}\big|_{z=0} = -C_h\, |\mathbf{U}|\, (\theta_{\text{sfc}} - \theta)
$$

**Humidity** — Robin condition:
$$
\overline{w'q'}\big|_{z=0} = -C_e\, |\mathbf{U}|\, (q_{\text{sfc}} - q)
$$

**TKE** — Dirichlet:
$$
e\big|_{z=0} = \frac{u_*^2}{\sqrt{C_m\, C_\varepsilon}}
$$

where $u_*^2 = C_d\, |\mathbf{U}|^2$ is the friction velocity squared.

### 6.2 Domain Top (z = H)

Two options:
- **Neumann** (default for idealized): zero-flux for all variables
- **Dirichlet / relaxation**: nudging toward large-scale values (for realistic forcings)

## 7. Coriolis Treatment

### 7.1 Forward-Backward Scheme (default)

Alternating update order on even/odd time steps:
- Even steps: advance $u$ first (explicit Coriolis), then $v$ using updated $u$
- Odd steps: advance $v$ first, then $u$ using updated $v$

This provides near-neutral stability for the inertial oscillation.

### 7.2 Semi-Implicit with Geostrophic Guide

$$
u^{n+1} = \frac{(1 - \gamma(1-\gamma)(f\Delta t)^2)\, u^n + f\Delta t\,(v^n - v_g) + (f\Delta t)^2\,\gamma\, v_g}{1 + \gamma^2 (f\Delta t)^2}
$$

with $\gamma = 0.55$ and analogous formula for $v^{n+1}$.

## 8. Nudging / Relaxation

Height-dependent Newtonian relaxation toward large-scale values $\phi_{\text{ref}}$:

$$
F^{\phi}_{\text{nudge}} = -\alpha(z)\, (\phi - \phi_{\text{ref}})
$$

where the relaxation coefficient $\alpha(z)$ follows a cubic profile in $\sigma = z/h_{\text{pbl}}$:

$$
\alpha(\sigma) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min})\, \sigma^3
$$

transitioning from weak relaxation below the PBL ($\alpha_{\min}$) to strong relaxation above ($\alpha_{\max}$).

## 9. Bulk Formulae

Transfer coefficients $C_d$, $C_h$, $C_e$ are computed from one of several bulk algorithms:
- NCAR (Large & Yeager 2004)
- COARE 3.0 (Fairall et al. 2003)
- COARE 3.5 (Edson et al. 2013)
- ECMWF (IFS documentation)

These depend on wind speed, air-sea temperature difference, and stability (roughness lengths). For idealized test cases, simplified constant or wind-speed-dependent formulations are used.

## 10. Physical Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| $g$ | 9.81 m s$^{-2}$ | Gravitational acceleration |
| $\beta$ | $g / \theta_{\text{ref}}$ | Buoyancy parameter |

## 11. TKE Closure Parameter Sets

| Parameter | CBR | CCH/MesoNH | Description |
|-----------|-----|-------------|-------------|
| $C_m$ | 0.0667 | 0.126 | Viscosity constant |
| $C_t$ | 0.1667 | 0.143 | Diffusivity constant |
| $C_e$ | 0.4 | 0.34 | TKE diffusion constant |
| $C_\varepsilon$ | 0.7 | 0.845 | TKE dissipation constant |
| $R_{ic}$ | 0.139 | 0.143 | Critical Richardson number |
| $R_{od}$ | 0.15 | 0.15 | Rodier shear parameter |
| $e_{\min}$ | 10$^{-6}$ | 10$^{-6}$ | Minimum TKE (m$^2$ s$^{-2}$) |
| $K_m^{\text{bak}}$ | 10$^{-4}$ | 10$^{-4}$ | Background viscosity (m$^2$ s$^{-1}$) |
| $K_t^{\text{bak}}$ | 10$^{-5}$ | 10$^{-5}$ | Background diffusivity (m$^2$ s$^{-1}$) |
