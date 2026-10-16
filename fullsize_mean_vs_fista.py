#!/usr/bin/env python3
"""Compare mean filling and DT-FISTA on full-resolution data.

NPZ inputs are read from ``data/inputs/``. Figures and
numerical products are written to ``figures/`` and ``data/<pulsar>/``.
"""

import os
import sys

_NUM_THREADS = "4"
os.environ["OMP_NUM_THREADS"]       = _NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"]  = _NUM_THREADS
os.environ["MKL_NUM_THREADS"]       = _NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"]   = _NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = _NUM_THREADS

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd
import fista_inpaint


def main(pulsar='B2021+51'):
    print(f"[{pulsar}] Starting the full-size mean-fill and FISTA comparison.")

    # 1. Load the dynamic spectrum and RFI mask.
    dyn, mask, ts, freqs = cd.load_all(pulsar)
    Nf, Nt = dyn.shape

    valid_idx = (mask > 0.5)
    invalid_idx = ~valid_idx
    n_invalid = int(np.sum(invalid_idx))

    # 2. Construct the mean-filled baseline.
    mean_val = np.mean(dyn[valid_idx])
    dyn_mean = dyn.copy()
    dyn_mean[invalid_idx] = mean_val

    # 3. Reuse cached FISTA output because full-resolution reconstruction is slow.
    npz_path = os.path.join(cd.pulsar_outdir(pulsar), 'fullsize_results.npz')
    dt_fista = None
    dyn_fista = None
    if os.path.exists(npz_path) and '--recompute' not in sys.argv:
        try:
            _d = np.load(npz_path)
            if ('dyn_fista' in _d.files and _d['dyn_fista'].shape == dyn.shape):
                dyn_fista = _d['dyn_fista']
        except Exception:
            dyn_fista = None
    if dyn_fista is not None:
        print(f"[{pulsar}] Using the cached FISTA reconstruction.")
    else:
        print(f"[{pulsar}] Running the FISTA reconstruction.")
        t0 = time.time()
        dyn_fista = fista_inpaint.fista_inpaint(dyn_mean, mask, verbose=False)
        dt_fista = time.time() - t0

    # 4. Compute the secondary spectra.
    hann_win = cd.make_hann_2d(Nf, Nt)
    sec_mean  = cd.secondary_spectrum_db(dyn_mean,  hann_win)
    sec_fista = cd.secondary_spectrum_db(dyn_fista, hann_win)

    # Save numerical products for reproducibility and downstream plotting.
    np.savez(npz_path, dyn_mean=dyn_mean, dyn_fista=dyn_fista,
             sec_mean_db=sec_mean, sec_fista_db=sec_fista,
             mask=mask, ts=ts, freqs=freqs)

    # Write a compact, human-readable record of the run configuration.
    info_txt = os.path.join(cd.pulsar_outdir(pulsar), 'fullsize_reconstruction_summary.txt')
    with open(info_txt, 'w', encoding='utf-8') as ft:
        ft.write(f"Full-Size Reconstruction Summary — {pulsar}\n"
                 f"{'='*60}\n"
                 f"Full-size dynamic spectrum: {Nf}×{Nt} (Nf×Nt)\n"
                 f"Frequency: {freqs[0]:.2f}–{freqs[-1]:.2f} MHz, "
                 f"Time: {ts[0]:.2f}–{ts[-1]:.2f} min\n"
                 f"Masked pixels: {n_invalid} ({n_invalid/(Nf*Nt)*100:.2f}%)\n"
                 f"Mean fill value: {mean_val:.4f}\n"
                 f"FISTA: default params (k=0.95)"
                 + (f", elapsed {dt_fista:.1f}s" if dt_fista else ", result cached")
                 + "\n")
    print(
        f"[{pulsar}] Full-size reconstruction analysis complete. "
        f"Data: {npz_path}; summary: {info_txt}.")
    return {
        'dyn_mean': dyn_mean, 'dyn_fista': dyn_fista,
        'sec_mean': sec_mean, 'sec_fista': sec_fista,
        'ts': ts, 'freqs': freqs,
    }


def plot_combined(results):
    """Plot both pulsars, using adjacent dynamic- and secondary-spectrum rows."""
    ordered = cd.ordered_pulsar_results(results, 'Full-size reconstruction analysis')
    fig, axes = plt.subplots(
        4, 2, figsize=(16, 20), sharey='row', constrained_layout=True,
        gridspec_kw={'wspace': 0.0, 'hspace': 0.08})
    fig.set_constrained_layout_pads(w_pad=0.01, wspace=0.0)

    for pulsar_index, (pulsar, result) in enumerate(ordered):
        dyn_row = 2 * pulsar_index
        sec_row = dyn_row + 1
        dyn_mean = result['dyn_mean']
        dyn_fista = result['dyn_fista']
        sec_mean = result['sec_mean']
        sec_fista = result['sec_fista']
        ts = result['ts']
        freqs = result['freqs']
        Nf, Nt = dyn_mean.shape

        # Scale each pulsar independently while applying one scale across methods.
        vmin, vmax = np.nanpercentile(dyn_mean, [0.5, 99.5])
        vmin = 0.0
        sec_vmin, sec_vmax = np.nanpercentile(sec_fista.ravel(), [50, 99.9])
        sec_vmin_disp = sec_vmin - 5

        ts_s = ts * 60.0
        ft_edges, t_edges = cd.make_sec_edges(Nf, Nt, freqs, ts_s)

        dyn_arrs = [dyn_mean, dyn_fista]
        dyn_titles = ["Mean-Filled Dynamic Spectrum", "FISTA-Filled Dynamic Spectrum"]
        for col, (arr, title) in enumerate(zip(dyn_arrs, dyn_titles)):
            ax = axes[dyn_row, col]
            im_dyn = ax.imshow(
                arr, origin='lower', aspect='auto', cmap='viridis',
                extent=[ts[0], ts[-1], freqs[0], freqs[-1]],
                norm=Normalize(vmin=vmin, vmax=vmax))
            ax.set_title(f"[{pulsar}] {title}", fontsize=14)
            ax.set_xlabel("Time (min)", fontsize=13)
            if col == 0:
                ax.set_ylabel("Freq (MHz)", fontsize=13)
        cbar_dyn = fig.colorbar(
            im_dyn, ax=axes[dyn_row, :].tolist(), fraction=0.025, pad=0.02,
            label="Intensity")
        cd.style_colorbar(cbar_dyn)

        sec_arrs = [sec_mean, sec_fista]
        sec_titles = ["Mean-Filled Secondary Spectrum",
                      "FISTA-Filled Secondary Spectrum"]
        for col, (arr, title) in enumerate(zip(sec_arrs, sec_titles)):
            ax = axes[sec_row, col]
            im_sec = ax.imshow(
                arr, origin='lower', aspect='auto', cmap='inferno',
                extent=[t_edges[0], t_edges[-1], ft_edges[0], ft_edges[-1]],
                norm=Normalize(vmin=sec_vmin_disp, vmax=sec_vmax))
            ax.set_title(f"[{pulsar}] {title}", fontsize=14)
            ax.set_xlabel("Doppler (mHz)", fontsize=13)
            if col == 0:
                ax.set_ylabel(r"Delay ($\mu$s)", fontsize=13)
        cbar_sec = fig.colorbar(
            im_sec, ax=axes[sec_row, :].tolist(), fraction=0.025, pad=0.02,
            label="dB")
        cd.style_colorbar(cbar_sec)

    cd.style_subplot_grid(axes)
    fig.suptitle(
        "Comparison of Mean and FISTA Inpainting in Dynamic and Secondary Spectra",
        fontsize=18)
    out_png = cd.combined_figure_path('fullsize_mean_vs_fista.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Combined full-size reconstruction figure saved to {out_png}.")


if __name__ == "__main__":
    _results = {}
    for _psr in cd.PULSAR_LIST:
        try:
            _results[_psr] = main(_psr)
        except Exception as e:
            print(f"Warning: full-size reconstruction analysis failed for {_psr}: {e}")
    plot_combined(_results)
