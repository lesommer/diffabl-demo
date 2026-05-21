"""ABL state and parameters."""

import math
import jax.numpy as jnp
import equinox as eqx
from typing import Literal


class ABLParams(eqx.Module):
    Cm: float = 0.0667
    Ct: float = 0.1667
    Ce: float = 0.4
    Ceps: float = 0.7
    Ric: float = 0.139
    Rod: float = 0.15
    Cek: float = 258.0
    tke_min: float = 1e-6
    avm_bak: float = 1e-4
    avt_bak: float = 1e-5
    phi_max: float = (1.0 - 2.2) / 2.2
    eps_sfc: float = 1.0 / (1.0 + 2.8 ** 2)
    nn_amxl: int = 1
    dt: float = 60.0
    f: float = 1e-4
    g: float = 9.81
    T_ref: float = 283.0
    gamma_cor: float = 0.55
    ln_geos_winds: bool = False
    SemiImp_Cor: bool = False
    vkarmn: float = 0.4
    mxl_min: float = 0.0
    rn_Lsfc: float = 0.0
    rn_Esfc: float = 0.0

    def __post_init__(self):
        if self.mxl_min == 0.0:
            object.__setattr__(self, 'mxl_min',
                               self.avm_bak / self.Cm / math.sqrt(self.tke_min))
        if self.rn_Lsfc == 0.0:
            object.__setattr__(self, 'rn_Lsfc',
                               self.vkarmn * math.sqrt(math.sqrt(self.Cm * self.Ceps)) / self.Cm)
        if self.rn_Esfc == 0.0:
            object.__setattr__(self, 'rn_Esfc',
                               1.0 / math.sqrt(self.Cm * self.Ceps))


def cbr_params(**overrides) -> ABLParams:
    defaults = dict(Cm=0.0667, Ct=0.1667, Ce=0.4, Ceps=0.7, Ric=0.139)
    defaults.update(overrides)
    return ABLParams(**defaults)


def cch_params(**overrides) -> ABLParams:
    defaults = dict(Cm=0.126, Ct=0.143, Ce=0.34, Ceps=0.845, Ric=0.143)
    defaults.update(overrides)
    return ABLParams(**defaults)


class ABLState(eqx.Module):
    u: jnp.ndarray
    v: jnp.ndarray
    theta: jnp.ndarray
    q: jnp.ndarray
    tke: jnp.ndarray
    avm: jnp.ndarray
    avt: jnp.ndarray
    mxlm: jnp.ndarray
    mxld: jnp.ndarray
    pblh: jnp.ndarray
