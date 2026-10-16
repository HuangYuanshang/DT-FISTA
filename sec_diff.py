#!/usr/bin/env python3
"""Plot secondary-spectrum residuals from saved patch-comparison products.

Run ``patch_compare.py`` first to create ``data/<pulsar>/patch_data.npz``
for both pulsars. Each secondary spectrum is normalized by its own DC power
before computing ``reconstructed_dB - original_dB``.

Outputs: ``figures/sec_diff.png`` and the residual arrays and text summaries
in ``data/<pulsar>/``.
"""

import os
import sys

_NUM_THREADS = "4"
os.environ["OMP_NUM_THREADS"]       = _NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"]  = _NUM_THREADS
os.environ["MKL_NUM_THREADS"]       = _NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"]   = _NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = _NUM_THREADS

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd


def main(pulsar='B2021+51'):
    print(f"[{pulsar}] Starting the secondary-spectrum residual analysis.")

    # Load the patch and reconstructions saved by patch_compare.py.
    npz_path = os.path.join(cd.pulsar_outdir(pulsar), 'patch_data.npz')
    d = np.load(npz_path)
    dyn_crop   = d['dyn_crop']
    dyn_fista  = d['dyn_fista']
    dyn_wiener = d['dyn_wiener']
    ts_crop    = d['ts_crop']
    freqs_crop = d['freqs_crop']
    f0, f1, t0, t1 = d['crop_region']
    N = int(d['crop_size'])

    # Apply the Hann taper and compute secondary-spectrum power.
    dyn_hole = d['dyn_hole']
    hann_win = cd.make_hann_2d(N, N)
    sec_orig_p  = cd.secondary_spectrum_power(dyn_crop,   hann_win)
    sec_hole_p  = cd.secondary_spectrum_power(dyn_hole,   hann_win)
    sec_fista_p = cd.secondary_spectrum_power(dyn_fista,  hann_win)
    sec_wien_p  = cd.secondary_spectrum_power(dyn_wiener, hann_win)

    # Normalize each spectrum by its own DC power before subtraction in dB.
    sec_orig_db  = cd.power_to_db_center(sec_orig_p)
    sec_hole_db  = cd.power_to_db_center(sec_hole_p)
    sec_fista_db = cd.power_to_db_center(sec_fista_p)
    sec_wien_db  = cd.power_to_db_center(sec_wien_p)

    diff_wien  = sec_wien_db  - sec_orig_db
    diff_fista = sec_fista_db - sec_orig_db
    diff_hole  = sec_hole_db  - sec_orig_db

    # Report the residual range for diagnostics only.
    print(
        f"[{pulsar}] Maximum absolute residual: "
        f"Mean-Original={np.nanmax(np.abs(diff_hole)):.2f} dB, "
        f"Wiener-Original={np.nanmax(np.abs(diff_wien)):.2f} dB, "
        f"FISTA-Original={np.nanmax(np.abs(diff_fista)):.2f} dB.")

    # Save residual arrays for reproducibility.
    residual_path = os.path.join(
        cd.pulsar_outdir(pulsar), 'secondary_spectrum_residuals.npz')
    np.savez(residual_path,
             diff_wien_db=diff_wien, diff_fista_db=diff_fista,
             diff_hole_db=diff_hole,
             sec_orig_db=sec_orig_db, sec_wien_db=sec_wien_db, sec_fista_db=sec_fista_db,
             sec_hole_db=sec_hole_db)

    # Write a compact provenance summary.
    info_txt = os.path.join(cd.pulsar_outdir(pulsar), 'secondary_spectrum_residuals_summary.txt')
    with open(info_txt, 'w', encoding='utf-8') as ft:
        ft.write(f"Secondary-Spectrum Residual Summary — {pulsar}\n"
                 f"{'='*60}\n"
                 f"Patch: {N}×{N} at [f {f0}:{f1}, t {t0}:{t1}], "
                 f"3 holes each {float(d['hole_frac'])*100:.0f}%\n"
                 f"Difference definition: each secondary spectrum converted to dB "
                 f"(its own (0,0) pixel = 0 dB), then subtracted\n"
                 f"diff |ΔdB| max: Mean−Orig = {np.nanmax(np.abs(diff_hole)):.2f} dB, "
                 f"Wiener−Orig = {np.nanmax(np.abs(diff_wien)):.2f} dB, "
                 f"FISTA−Orig = {np.nanmax(np.abs(diff_fista)):.2f} dB\n")
    print(
        f"[{pulsar}] Secondary-spectrum residual analysis complete. "
        f"Data: {residual_path}; summary: {info_txt}.")
    return {
        'diff_hole': diff_hole, 'diff_wien': diff_wien,
        'diff_fista': diff_fista, 'ts_crop': ts_crop,
        'freqs_crop': freqs_crop, 'N': N,
    }


def plot_combined(results):
    """Plot one residual row and one shared colorbar per pulsar."""
    ordered = cd.ordered_pulsar_results(results, 'Secondary-spectrum residual analysis')
    fig, axes = plt.subplots(
        2, 3, figsize=(15.5, 12), sharey='row', constrained_layout=True,
        gridspec_kw={'wspace': 0.0, 'hspace': 0.08})
    fig.set_constrained_layout_pads(w_pad=0.01, wspace=0.0)

    diff_titles = ["Mean-filled minus original", "Wiener minus original",
                   "DT-FISTA minus original"]
    for row, (pulsar, result) in enumerate(ordered):
        ts_crop_s = result['ts_crop'] * 60.0
        ft_edges, t_edges = cd.make_sec_edges(
            result['N'], result['N'], result['freqs_crop'], ts_crop_s)

        # Set each row's symmetric scale from its FISTA residual distribution.
        diff_fista = result['diff_fista']
        vmax_abs = np.nanpercentile(
            np.abs(diff_fista[np.isfinite(diff_fista)].ravel()), 99.5)
        norm = Normalize(vmin=-vmax_abs, vmax=vmax_abs)
        diff_arrs = [result['diff_hole'], result['diff_wien'], diff_fista]

        for col, (diff, title) in enumerate(zip(diff_arrs, diff_titles)):
            ax = axes[row, col]
            pcm = ax.pcolormesh(
                t_edges, ft_edges, diff, shading='auto', cmap='RdBu_r', norm=norm)
            ax.set_title(f"[{pulsar}] {title}", fontsize=14)
            ax.set_box_aspect(1)
            ax.set_xlabel("Doppler (mHz)", fontsize=13)
            if col == 0:
                ax.set_ylabel(r"Delay ($\mu$s)", fontsize=13)
        cbar = fig.colorbar(
            pcm, ax=axes[row, :].tolist(), fraction=0.022, pad=0.02,
            shrink=0.60, label="Δ dB")
        cd.style_colorbar(cbar)

    cd.style_subplot_grid(axes, panel_labelsize=14)
    fig.suptitle(
        "Secondary-Spectrum Residuals for Mean, Wiener, and DT-FISTA Inpainting",
        fontsize=18)
    out_png = cd.combined_figure_path('sec_diff.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Combined secondary-spectrum residual figure saved to {out_png}.")


if __name__ == "__main__":
    _results = {}
    for _psr in cd.PULSAR_LIST:
        try:
            _results[_psr] = main(_psr)
        except Exception as e:
            print(f"Warning: secondary-spectrum residual analysis failed for {_psr}: {e}")
    plot_combined(_results)
