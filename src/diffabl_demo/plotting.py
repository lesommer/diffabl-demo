"""Plotting utilities for the ABL demo cases.

Generates figures matching Lemarié et al. (2021):
  - Fig. 4: Andren 1994 Ekman spiral (wind hodograph)
  - Fig. 5: Cuxart 2005 convective ABL (vertical profiles)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from diffabl_demo.grid import ABLGrid
from diffabl_demo.state import ABLState


def plot_andren94_hodograph(
    state: ABLState,
    grid: ABLGrid,
    save_path: str = "andren94_hodograph.png",
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    u = np.array(state.u)
    v = np.array(state.v)
    z = np.array(grid.ght)
    ax.plot(u[1:], v[1:], "b-o", markersize=3, linewidth=1)
    ax.plot(u[1], v[1], "rs", markersize=8, label="Surface")
    ax.plot(u[-1], v[-1], "k^", markersize=8, label="Top")
    ax.set_xlabel("u (m/s)")
    ax.set_ylabel("v (m/s)")
    ax.set_title("Andren 1994 Ekman Spiral — Wind Hodograph")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_andren94_profiles(
    state: ABLState,
    grid: ABLGrid,
    save_path: str = "andren94_profiles.png",
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    u = np.array(state.u)
    v = np.array(state.v)
    tke = np.array(state.tke)
    avm = np.array(state.avm)
    z = np.array(grid.ght)

    axes[0].plot(u[1:], z[1:], "b-", linewidth=1.5, label="u")
    axes[0].plot(v[1:], z[1:], "r-", linewidth=1.5, label="v")
    axes[0].set_xlabel("Wind speed (m/s)")
    axes[0].set_ylabel("Height (m)")
    axes[0].set_title("Wind profiles")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tke[1:], z[1:], "g-", linewidth=1.5)
    axes[1].set_xlabel("TKE (m²/s²)")
    axes[1].set_ylabel("Height (m)")
    axes[1].set_title("TKE profile")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(avm[1:], z[1:], "m-", linewidth=1.5)
    axes[2].set_xlabel("Km (m²/s)")
    axes[2].set_ylabel("Height (m)")
    axes[2].set_title("Eddy viscosity")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Andren 1994 Ekman Spiral", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_cuxart05_profiles(
    state: ABLState,
    grid: ABLGrid,
    save_path: str = "cuxart05_profiles.png",
):
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    theta = np.array(state.theta)
    q = np.array(state.q)
    tke = np.array(state.tke)
    avm = np.array(state.avm)
    u = np.array(state.u)
    z = np.array(grid.ght)

    axes[0].plot(theta[1:], z[1:], "r-", linewidth=1.5)
    axes[0].set_xlabel("Potential temp (K)")
    axes[0].set_ylabel("Height (m)")
    axes[0].set_title("θ profile")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(u[1:], z[1:], "b-", linewidth=1.5, label="u")
    axes[1].set_xlabel("Wind speed (m/s)")
    axes[1].set_ylabel("Height (m)")
    axes[1].set_title("Wind profile")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(tke[1:], z[1:], "g-", linewidth=1.5)
    axes[2].set_xlabel("TKE (m²/s²)")
    axes[2].set_ylabel("Height (m)")
    axes[2].set_title("TKE profile")
    axes[2].grid(True, alpha=0.3)

    axes[3].semilogx(avm[1:], z[1:], "m-", linewidth=1.5)
    axes[3].set_xlabel("Km (m²/s)")
    axes[3].set_ylabel("Height (m)")
    axes[3].set_title("Eddy viscosity")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Cuxart 2005 Convective ABL", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    from diffabl_demo.demos import andren94, cuxart05
    from diffabl_demo.state import cbr_params, cch_params

    print("Running Andren 1994...")
    state_a, g_a = andren94(cbr_params(dt=60.0, f=1e-4), n_steps=1670)
    plot_andren94_hodograph(state_a, g_a)
    plot_andren94_profiles(state_a, g_a)

    print("Running Cuxart 2005...")
    state_c, g_c = cuxart05(cch_params(dt=10.0, f=1.39e-4), n_steps=3240)
    plot_cuxart05_profiles(state_c, g_c)

    print("Done.")
