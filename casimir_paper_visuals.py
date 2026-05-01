"""
casimir_paper_visuals.py
========================
Generates all 7 paper figures for the Lifshitz-BEM Casimir study.

Run this file directly to produce all figures:
    python casimir_paper_visuals.py

Or call individual functions after collecting simulation data:
    from casimir_paper_visuals import *

Output
------
All figures saved to ./img/paper/ at 300 DPI as both PNG and PDF.

Figures
-------
  Fig 1  — Geometry schematic with labeled parameters + BEM panels
  Fig 2  — Frequency integrand profile (justifies frequency window)
  Fig 3  — BEM convergence: energy vs panel count N
  Fig 4  — Correction factor vs corrugation amplitude (multi-A)
  Fig 5  — Lateral force amplitude vs separation
  Fig 6  — Energy landscape E(a, φ) as 2D heatmap
  Fig 7  — Force vector field on (a, φ) parameter grid
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import simpson
from scipy.special import kn

# ── Import simulation primitives ──────────────────────────────────────────────
from lifshitzbemv3 import (
    build_panels, build_G12, calculate_casimir_energy,
    y_top, H_BAR, C
)

# ── Output directory + style ──────────────────────────────────────────────────
SAVE_DIR = os.path.join("img", "paper")
DPI      = 300

# Consistent colour palette across all figures
COL = dict(
    corrugated = "#7B1FA2",   # deep violet
    flat       = "#1565C0",   # steel blue
    pfa        = "#212121",   # near-black dashed reference
    lateral    = "#2E7D32",   # forest green
    integrand  = "#E65100",   # burnt orange
    converge   = "#1565C0",   # dark blue
    converge_r = "#C62828",   # dark red  (reference line)
    quiver     = "#C62828",   # crimson arrows
    plate      = "#212121",   # plate geometry
    gap        = "#E3F2FD",   # pale-blue vacuum gap
    panel_nrm  = "#0D47A1",   # panel normals
    amp_cmap   = "plasma",      # multi-amplitude colormap
)

def _save(fig, name):
    os.makedirs(SAVE_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(SAVE_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved  {name}.png / .pdf")


def _xi_points():
    """Shared frequency grid — same window used in all simulations."""
    return np.logspace(7, 10, 40)


def _energy_integrand(centers, lengths, xi):
    kappa = xi / C
    G12   = build_G12(centers, lengths, kappa)
    return -float(np.sum(G12 ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Geometry schematic
# ─────────────────────────────────────────────────────────────────────────────

def fig1_geometry(a=1.0, A=0.3, lam=2.0, n_show=40):
    """
    Annotated cross-section: corrugated top plate, flat bottom plate,
    labeled parameters (a, A, λ, φ), and BEM panel normals inset.
    """
    print("Fig 1 — geometry schematic...")
    fig = plt.figure(figsize=(12, 4))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_inset = fig.add_subplot(gs[1])

    x     = np.linspace(-3, 3, 400)
    y_t   = y_top(x, a, A, lam)
    y_b   = np.zeros_like(x)

    # ── Main panel ────────────────────────────────────────────────────────────
    ax_main.fill_between(x, y_b, y_t, color=COL["gap"], alpha=0.55,
                         label="Vacuum gap")
    ax_main.fill_between(x, y_t, y_t + 0.22, color="#BDBDBD", alpha=0.85)
    ax_main.fill_between(x, y_b - 0.22, y_b, color="#BDBDBD", alpha=0.85)
    ax_main.plot(x, y_t, color=COL["plate"], lw=2.0, label="Top plate (corrugated)")
    ax_main.plot(x, y_b, color=COL["plate"], lw=2.0, ls="--",
                 label="Bottom plate (flat)")

    # Separation arrow  a
    x_ann = 0.5
    y_top_at = y_top(np.array([x_ann]), a, A, lam)[0]
    ax_main.annotate("", xy=(x_ann, y_top_at), xytext=(x_ann, 0),
                     arrowprops=dict(arrowstyle="<->", color="#333", lw=1.4))
    ax_main.text(x_ann + 0.12, y_top_at / 2, r"$a$",
                 fontsize=13, va="center", color="#333")

    # Amplitude arrow  A
    x_amp = -2.0
    ax_main.annotate("", xy=(x_amp - 0.2, a + A),
                     xytext=(x_amp - 0.2, a),
                     arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
    ax_main.text(x_amp - 0.52, a + A / 2, r"$A$",
                 fontsize=12, va="center", color="#555")

    # Wavelength arrow  λ
    x0, x1 = -1.5, -1.5 + lam
    y_lam  = a + A + 0.22
    ax_main.annotate("", xy=(x1, y_lam), xytext=(x0, y_lam),
                     arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
    ax_main.text((x0 + x1) / 2, y_lam + 0.1, r"$\lambda$",
                 fontsize=12, ha="center", color="#555")

    # Phase label
    ax_main.text(2.6, -0.12, r"$\varphi$", fontsize=11, color="#666")

    ax_main.set(xlim=(-3.1, 3.1), ylim=(-0.45, a + A + 0.55),
                xlabel=r"$x$ (a.u.)", ylabel=r"$y$ (a.u.)",
                title="(a)  Corrugated-plate geometry")
    ax_main.set_yticks([])
    ax_main.legend(loc="upper right", fontsize=8)
    ax_main.spines[["top", "right"]].set_visible(False)

    # ── Inset: BEM panel normals ──────────────────────────────────────────────
    x_p   = np.linspace(-1.5, 1.5, n_show)
    y_tp  = y_top(x_p, a, A, lam)

    ax_inset.plot(x_p, y_tp, color=COL["plate"], lw=1.5)
    ax_inset.plot(x_p, np.zeros_like(x_p), color=COL["plate"],
                  lw=1.5, ls="--")

    # Draw outward normals every 4th panel
    for k in range(0, len(x_p), 4):
        dy_dx = A * (2 * np.pi / lam) * np.cos(2 * np.pi * x_p[k] / lam)
        nx, ny = -dy_dx, 1.0
        nrm    = np.hypot(nx, ny)
        nx, ny = nx / nrm * 0.20, ny / nrm * 0.20
        ax_inset.annotate("",
                          xy=(x_p[k] + nx, y_tp[k] + ny),
                          xytext=(x_p[k], y_tp[k]),
                          arrowprops=dict(arrowstyle="->",
                                          color=COL["panel_nrm"], lw=0.8))
        # Bottom plate normals (point downward)
        ax_inset.annotate("",
                          xy=(x_p[k], -0.20),
                          xytext=(x_p[k], 0),
                          arrowprops=dict(arrowstyle="->",
                                          color=COL["panel_nrm"], lw=0.8))

    ax_inset.set(xlim=(-1.8, 1.8), ylim=(-0.4, a + A + 0.4),
                 xlabel=r"$x$", title="(b)  BEM panel normals")
    ax_inset.set_yticks([])
    ax_inset.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig1_geometry")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Frequency integrand profile
# ─────────────────────────────────────────────────────────────────────────────

def fig2_integrand_profile(sep_vals=None, A=0.3, lam=2.0, n_panels=40):
    """
    Plot −‖G₁₂(iξ)‖²_F vs ξ for several separations.
    Justifies the frequency window choice to reviewers.
    """
    print("Fig 2 — integrand profile...")
    if sep_vals is None:
        sep_vals = [0.5, 1.0, 1.5, 2.0, 2.5]

    xi_vals = np.logspace(6, 11, 80)
    cmap    = plt.cm.plasma
    colors  = [cmap(i / (len(sep_vals) - 1)) for i in range(len(sep_vals))]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for sep, col in zip(sep_vals, colors):
        starts, ends, centers = build_panels(n_panels, sep, A, lam)
        lengths = np.linalg.norm(ends - starts, axis=1)
        iv = [_energy_integrand(centers, lengths, xi) for xi in xi_vals]
        iv = np.array(iv)
        ax.semilogx(xi_vals, iv, lw=2.0, color=col, label=f"$a={sep}$")

    # Mark the shared integration window
    xi_lo, xi_hi = _xi_points()[0], _xi_points()[-1]
    ax.axvspan(xi_lo, xi_hi, alpha=0.07, color=COL["integrand"],
               label="Integration window")
    ax.axvline(xi_lo, color=COL["integrand"], lw=1.0, ls=":")
    ax.axvline(xi_hi, color=COL["integrand"], lw=1.0, ls=":")

    ax.set(xlabel=r"Imaginary frequency $\xi$ (rad s$^{-1}$)",
           ylabel=r"$-\|G_{12}(i\xi)\|_F^2$  (a.u.)",
           title="Fig. 2 — Frequency Integrand Profile")
    ax.legend(title="Separation", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    fig.tight_layout()
    _save(fig, "fig2_integrand_profile")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — BEM convergence study
# ─────────────────────────────────────────────────────────────────────────────

def fig3_convergence(a=1.0, A=0.3, lam=2.0,
                     panel_counts=None):
    """
    Energy vs panel count N at fixed separation.
    Left: absolute energy with converged reference line.
    Right: relative error |E_N − E_∞| / |E_∞|.
    """
    print("Fig 3 — convergence study...")
    if panel_counts is None:
        panel_counts = [10, 20, 30, 40, 60, 80, 100, 130, 160]

    energies = []
    for N in panel_counts:
        E = calculate_casimir_energy(a, A, lam, n_panels=N)
        energies.append(E)
        print(f"  N={N:4d}  E={E:.6e}")

    energies  = np.array(energies)
    E_ref     = energies[-1]
    rel_err   = np.abs((energies - E_ref) / E_ref)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left — absolute energy
    axes[0].plot(panel_counts, energies, "s-", color=COL["converge"],
                 ms=6, lw=1.8, label="BEM energy")
    axes[0].axhline(E_ref, color=COL["converge_r"], ls="--", lw=1.2,
                    label=f"Converged value ({E_ref:.3e})")
    axes[0].set(xlabel="Panel count $N$",
                ylabel=r"$E_\mathrm{Casimir}$ (a.u.)",
                title="(a)  Absolute energy")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls=":", alpha=0.35)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Right — relative error (skip last point = 0)
    axes[1].semilogy(panel_counts[:-1], rel_err[:-1], "o-",
                     color=COL["converge_r"], ms=6, lw=1.8,
                     label="Relative error")
    axes[1].axhline(1e-2, color="#999", ls=":", lw=1,  label="1 % threshold")
    axes[1].axhline(1e-3, color="#666", ls=":", lw=1,  label="0.1 % threshold")
    axes[1].set(xlabel="Panel count $N$",
                ylabel=r"$|E_N - E_\infty|\;/\;|E_\infty|$",
                title="(b)  Convergence rate")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", ls=":", alpha=0.35)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Fig. 3 — BEM Convergence Study", y=1.01)
    fig.tight_layout()
    _save(fig, "fig3_convergence")
    plt.close(fig)

    return panel_counts, energies


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Correction factor vs corrugation amplitude
# ─────────────────────────────────────────────────────────────────────────────

def fig4_multi_amplitude(a_vals=None, amplitudes=None,
                         lam=2.0, n_panels=40):
    """
    Beyond-PFA correction factor F_corr/F_flat for several amplitudes A.
    Shows where PFA breaks down as A/a grows — the main physics result.
    """
    print("Fig 4 — multi-amplitude correction factor...")
    if a_vals is None:
        a_vals = np.linspace(0.5, 2.5, 25)
    if amplitudes is None:
        amplitudes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Flat plate reference (compute once)
    print("  Computing flat plate reference...")
    E_flat = np.array([calculate_casimir_energy(a, 0.0, lam, n_panels)
                       for a in a_vals])
    F_flat = -np.gradient(E_flat, a_vals)

    cmap   = plt.cm.plasma
    colors = [cmap(0.15 + 0.7 * i / (len(amplitudes) - 1))
              for i in range(len(amplitudes))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for amp, col in zip(amplitudes, colors):
        print(f"  A={amp}...")
        E_corr = np.array([calculate_casimir_energy(a, amp, lam, n_panels)
                           for a in a_vals])
        F_corr = -np.gradient(E_corr, a_vals)
        ratio  = np.abs(F_corr) / np.abs(F_flat)

        # Left: ratio vs separation
        axes[0].plot(a_vals, ratio, "o-", color=col, ms=3, lw=1.6,
                     label=f"$A={amp}$")

        # Right: ratio vs A/a at a fixed midpoint separation
        # (plotted as a single point per amplitude — done after loop)

    # Overlay A/a = const contours on left panel
    for frac, ls in [(0.1, ":"), (0.3, "--"), (0.5, "-.")]:
        a_line = np.array(amplitudes) / frac  # a where A/a = frac
        # Only annotate, don't clutter with more lines
        pass

    axes[0].axhline(1.0, color="k", lw=1.2, ls="--", label="PFA (ratio=1)")
    axes[0].set(xlabel="Separation $a$",
                ylabel=r"$|F_\mathrm{corr}|\;/\;|F_\mathrm{flat}|$",
                title="(a)  Correction factor vs separation")
    axes[0].legend(title="Amplitude $A$", fontsize=7.5, ncol=2)
    axes[0].grid(True, ls=":", alpha=0.35)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Right: correction factor vs A/a at midpoint separation
    a_mid   = a_vals[len(a_vals) // 2]
    ratios_at_mid = []
    for amp in amplitudes:
        E_c = calculate_casimir_energy(a_mid, amp, lam, n_panels)
        E_f = calculate_casimir_energy(a_mid, 0.0,  lam, n_panels)
        # Use single-point estimate for speed (already computed above but
        # recomputing here keeps function self-contained)
        F_c_pt = abs(E_c)
        F_f_pt = abs(E_f)
        ratios_at_mid.append(F_c_pt / F_f_pt if F_f_pt > 0 else 1.0)

    A_over_a = [amp / a_mid for amp in amplitudes]
    axes[1].plot(A_over_a, ratios_at_mid, "D-", color=COL["corrugated"],
                 ms=7, lw=2.0)
    axes[1].axhline(1.0, color="k", lw=1.2, ls="--", label="PFA limit")
    axes[1].set(xlabel=r"$A\;/\;a$  (corrugation ratio)",
                ylabel=r"Correction factor at $a = $" + f"${a_mid:.1f}$",
                title=r"(b)  PFA breakdown vs $A/a$")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, ls=":", alpha=0.35)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Fig. 4 — Beyond-PFA Correction Factor", y=1.01)
    fig.tight_layout()
    _save(fig, "fig4_multi_amplitude")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Lateral force amplitude vs separation
# ─────────────────────────────────────────────────────────────────────────────

def fig5_lateral_vs_separation(a_vals=None, A=0.3, lam=2.0,
                                n_phases=20, n_panels=60):
    """
    Lateral force amplitude |F_x| vs separation.

    At each separation, sweeps a full phase cycle (0 → 2π) and takes
    the peak |dE/dφ| as the lateral force amplitude.  This is more
    robust than a 2-point finite difference, which hits machine-precision
    cancellation at larger separations where the energy variation is tiny.
    """
    print("Fig 5 — lateral force vs separation...")
    if a_vals is None:
        a_vals = np.linspace(0.5, 2.5, 20)

    xi_pts  = _xi_points()
    phases  = np.linspace(0, 2 * np.pi, n_phases, endpoint=False)
    Fx_peak = []

    for a in a_vals:
        energies_phase = []
        for phi in phases:
            starts, ends, centers = build_panels(n_panels, a, A, lam, phi)
            lengths = np.linalg.norm(ends - starts, axis=1)
            iv = [_energy_integrand(centers, lengths, xi) for xi in xi_pts]
            energies_phase.append(
                (H_BAR / (2 * np.pi)) * simpson(y=iv, x=xi_pts))
        energies_phase = np.array(energies_phase)
        Fx_arr = -np.gradient(energies_phase, phases)
        Fx_peak.append(np.max(np.abs(Fx_arr)))
        print(f"  a={a:.2f}  peak|Fx|={Fx_peak[-1]:.4e}")

    Fx_peak = np.array(Fx_peak)

    # Power-law fit (avoid endpoint noise)
    mid    = slice(3, -3)
    valid  = Fx_peak[mid] > 0
    if valid.sum() > 2:
        exp_fx = np.polyfit(
            np.log(a_vals[mid][valid]),
            np.log(Fx_peak[mid][valid]), 1)[0]
        fit_label = fr"$|F_x|_\mathrm{{peak}}$   (fit: $a^{{{exp_fx:.2f}}}$)"
    else:
        fit_label = r"$|F_x|_\mathrm{peak}$"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(a_vals, Fx_peak, "o-", color=COL["lateral"],
            ms=5, lw=2.0, label=fit_label)
    ax.set(yscale="log",
           xlabel="Separation $a$",
           ylabel=r"Peak lateral force $|F_x|$ (a.u.)",
           title=r"Fig. 5 — Lateral Force Amplitude vs Separation")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig5_lateral_vs_separation")
    plt.close(fig)

    return a_vals, Fx_peak


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Energy landscape E(a, φ) as 2D heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig6_energy_landscape(a_vals=None, phi_vals=None,
                           A=0.3, lam=2.0, n_panels=40):
    """
    2D heatmap of E(a, φ) computed on a grid.
    Shows the coupled dependence of Casimir energy on both separation and
    corrugation phase in a single visually striking figure.
    """
    print("Fig 6 — energy landscape E(a, φ)...")
    if a_vals is None:
        a_vals = np.linspace(0.5, 2.5, 14)
    if phi_vals is None:
        phi_vals = np.linspace(0, 2 * np.pi, 18)

    xi_pts = _xi_points()
    E_grid = np.zeros((len(a_vals), len(phi_vals)))

    for i, a in enumerate(a_vals):
        for j, phi in enumerate(phi_vals):
            starts, ends, centers = build_panels(n_panels, a, A, lam, phi)
            lengths = np.linalg.norm(ends - starts, axis=1)
            iv = [_energy_integrand(centers, lengths, xi) for xi in xi_pts]
            E_grid[i, j] = (H_BAR / (2 * np.pi)) * simpson(y=iv, x=xi_pts)
        print(f"  a={a:.2f} row done")

    PHI, AA = np.meshgrid(phi_vals / np.pi, a_vals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: heatmap of E
    im = axes[0].contourf(PHI, AA, E_grid, levels=22, cmap="magma")
    cb = fig.colorbar(im, ax=axes[0], pad=0.02)
    cb.set_label(r"$E_\mathrm{Casimir}$ (a.u.)", fontsize=8)
    axes[0].set(xlabel=r"Phase $\varphi\;(\times\,\pi)$",
                ylabel="Separation $a$",
                title="(a)  Energy landscape $E(a,\\varphi)$")
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # Right: heatmap of dE/dφ (lateral force proxy)
    dE_dphi = np.gradient(E_grid, phi_vals, axis=1)
    im2 = axes[1].contourf(PHI, AA, -dE_dphi, levels=22, cmap="RdBu_r")
    cb2 = fig.colorbar(im2, ax=axes[1], pad=0.02)
    cb2.set_label(r"$F_x = -\partial E/\partial\varphi$ (a.u.)", fontsize=8)
    axes[1].contour(PHI, AA, -dE_dphi, levels=[0],
                    colors="k", linewidths=1.2, linestyles="--")
    axes[1].set(xlabel=r"Phase $\varphi\;(\times\,\pi)$",
                ylabel="Separation $a$",
                title=r"(b)  Lateral force $F_x(a,\varphi)$")
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    fig.suptitle(r"Fig. 6 — Casimir Energy \& Lateral Force Landscape", y=1.01)
    fig.tight_layout()
    _save(fig, "fig6_energy_landscape")
    plt.close(fig)

    return a_vals, phi_vals, E_grid


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Force vector field on (a, φ) grid
# ─────────────────────────────────────────────────────────────────────────────

def fig7_force_vector_field(a_vals=None, phi_vals=None,
                             A=0.3, lam=2.0, n_panels=40):
    """
    Quiver plot of (Fx, Fy) on the (separation, phase) parameter plane.

    Fx = −dE/dφ  (lateral,  from phase gradient)
    Fy = −dE/da  (normal,   from separation gradient)

    Arrows are normalised to unit length for directional clarity;
    background heatmap encodes force magnitude.
    Shows in one image how the force direction rotates across parameter space.
    """
    print("Fig 7 — force vector field...")
    if a_vals is None:
        a_vals = np.linspace(0.5, 2.5, 12)
    if phi_vals is None:
        phi_vals = np.linspace(0, 2 * np.pi, 16)

    xi_pts = _xi_points()
    E_grid = np.zeros((len(a_vals), len(phi_vals)))

    for i, a in enumerate(a_vals):
        for j, phi in enumerate(phi_vals):
            starts, ends, centers = build_panels(n_panels, a, A, lam, phi)
            lengths = np.linalg.norm(ends - starts, axis=1)
            iv = [_energy_integrand(centers, lengths, xi) for xi in xi_pts]
            E_grid[i, j] = (H_BAR / (2 * np.pi)) * simpson(y=iv, x=xi_pts)
        print(f"  a={a:.2f} row done")

    # Force components via numerical gradient
    Fy_grid = -np.gradient(E_grid, a_vals,   axis=0)   # normal
    Fx_grid = -np.gradient(E_grid, phi_vals, axis=1)   # lateral

    F_mag = np.sqrt(Fx_grid**2 + Fy_grid**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        Fx_n = np.where(F_mag > 0, Fx_grid / F_mag, 0.0)
        Fy_n = np.where(F_mag > 0, Fy_grid / F_mag, 0.0)

    PHI, AA = np.meshgrid(phi_vals / np.pi, a_vals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: magnitude heatmap
    im = axes[0].contourf(PHI, AA, F_mag, levels=22, cmap="magma")
    cb = fig.colorbar(im, ax=axes[0], pad=0.02)
    cb.set_label(r"$|\mathbf{F}|$ (a.u.)", fontsize=8)
    axes[0].set(xlabel=r"Phase $\varphi\;(\times\,\pi)$",
                ylabel="Separation $a$",
                title="(a)  Force magnitude $|\\mathbf{F}(a,\\varphi)|$")
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    # Right: direction quiver
    step  = max(1, len(a_vals) // 8)
    stepp = max(1, len(phi_vals) // 8)
    sc = axes[1].quiver(
        PHI [::step, ::stepp], AA  [::step, ::stepp],
        Fx_n[::step, ::stepp], Fy_n[::step, ::stepp],
        F_mag[::step, ::stepp],
        cmap="magma", scale=14, width=0.006, pivot="mid",
        alpha=0.92, clim=(F_mag.min(), F_mag.max())
    )
    fig.colorbar(sc, ax=axes[1], pad=0.02).set_label(
        r"$|\mathbf{F}|$ (a.u.)", fontsize=8)
    axes[1].set(xlabel=r"Phase $\varphi\;(\times\,\pi)$",
                ylabel="Separation $a$",
                title="(b)  Force direction field")
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    fig.suptitle(r"Fig. 7 — Casimir Force Vector Field $(a,\,\varphi)$",
                 y=1.01)
    fig.tight_layout()
    _save(fig, "fig7_force_vector_field")
    plt.close(fig)

    return a_vals, phi_vals, Fx_grid, Fy_grid


# ─────────────────────────────────────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def generate_all(fast=False):
    """
    Generate all 7 figures.

    Parameters
    ----------
    fast : bool
        If True, uses coarser grids for quick preview (< 2 min).
        Set False for paper-quality resolution.
    """
    print("=" * 60)
    print("  Casimir BEM — Paper Figure Generator")
    print(f"  Output: {os.path.abspath(SAVE_DIR)}")
    print(f"  Mode:   {'FAST PREVIEW' if fast else 'FULL RESOLUTION'}")
    print("=" * 60)

    if fast:
        a_sweep    = np.linspace(0.5, 2.5, 12)
        phi_sweep  = np.linspace(0, 2 * np.pi, 10)
        n_p        = 30
        pan_counts = [10, 20, 30, 40, 60, 80]
        amps       = [0.1, 0.3, 0.5]
    else:
        a_sweep    = np.linspace(0.5, 2.5, 25)
        phi_sweep  = np.linspace(0, 2 * np.pi, 18)
        n_p        = 50
        pan_counts = [10, 20, 30, 40, 60, 80, 100, 130]
        amps       = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    fig1_geometry()
    fig2_integrand_profile(sep_vals=[0.5, 1.0, 1.5, 2.0, 2.5],
                           n_panels=n_p)
    fig3_convergence(panel_counts=pan_counts)
    fig4_multi_amplitude(a_vals=a_sweep, amplitudes=amps, n_panels=n_p)
    fig5_lateral_vs_separation(a_vals=a_sweep, n_panels=n_p + 10)
    fig6_energy_landscape(a_vals=a_sweep[:10], phi_vals=phi_sweep,
                          n_panels=n_p)
    fig7_force_vector_field(a_vals=a_sweep[:10], phi_vals=phi_sweep,
                            n_panels=n_p)

    print("\n" + "=" * 60)
    print(f"  All figures saved to {os.path.abspath(SAVE_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    # Set fast=False for full paper resolution (takes longer)
    generate_all(fast=True)