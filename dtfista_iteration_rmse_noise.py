"""Record DT-FISTA error and noise metrics over the iterations.

Run from this directory with ``python dtfista_iteration_rmse_noise.py``.
NPZ inputs are read from ``data/inputs/`` and figures are written to ``figures/``.
"""
from pathlib import Path
import os
import time

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DATA / ".matplotlib"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common_data as cd
import fista_inpaint

N = 300
OUT = FIGURES
FILENAMES = {
    "B1237+25": "B1237+25.npz",
    "B2021+51": "B2021+51.npz",
}

def configure_data():
    """Resolve the two NPZ input files."""
    for psr, filename in FILENAMES.items():
        cd.PULSARS[psr]["input_file"] = str(DATA / "inputs" / filename)

def load_observation(psr):
    configure_data()
    path = DATA / "inputs" / FILENAMES[psr]
    if not path.is_file():
        raise FileNotFoundError(f"Place {FILENAMES[psr]} in data/inputs: {path}")
    return cd.load_all(psr)

def reference_patch(psr):
    """Select one reproducible reference patch."""
    dyn, mask, ts, freqs = load_observation(psr)
    tolerance = .01 if psr == "B1237+25" else 0.0
    f0, t0, count = cd.find_clean_patch_random(mask, N, seed=0,
                                              max_masked_frac=tolerance)
    patch = dyn[f0:f0+N, t0:t0+N].copy()
    rfi = mask[f0:f0+N, t0:t0+N] <= .5
    if rfi.any():
        patch = cd.fill_rfi_interp(patch, rfi)
    assert patch.shape == (N, N) and np.isfinite(patch).all()
    print(f"[{psr}] Reference at ({f0}, {t0}); interpolated {rfi.sum()} input-mask samples.", flush=True)
    return patch, ts[t0:t0+N], freqs[f0:f0+N], (f0,f0+N,t0,t0+N)

def compute_history(pulsar='B2021+51'):
    print(f"[{pulsar}] Starting the FISTA iteration-convergence analysis.")

    dyn_crop, ts, freqs, crop = reference_patch(pulsar)
    hole_mask, _ = cd.make_combined_hole(N, 4500, 0)
    dyn_hole = dyn_crop.copy()
    dyn_hole[~hole_mask] = dyn_crop[hole_mask].mean()
    f0, f1, t0, t1 = crop

    invalid_idx = (hole_mask < 0.5)
    n_hole = int(np.sum(invalid_idx))

    # Precompute quantities used by the callback.
    hann_win = cd.make_hann_2d(N, N)
    sec_orig_p = cd.secondary_spectrum_power(dyn_crop, hann_win)

    # Run FISTA and record each iteration.
    history = {'iter': [], 'dyn_rmse': [], 'sec_rmse': [], 'noise_db': [], 'pass_id': []}
    state = {'count': 0, 'pass': 1, 'last_it': 0}

    def record(it, recon):
        # Keep a monotonic counter across passes.
        if it <= state['last_it'] and state['count'] > 0:
            state['pass'] += 1          # Enter the edge-refinement pass.
        state['last_it'] = it
        state['count'] += 1
        dyn_rmse = cd.dyn_mask_rmse(recon, dyn_crop, invalid_idx)
        sec_p = cd.secondary_spectrum_power(recon, hann_win)
        sec_rmse = cd.sec_log10_rmse(sec_p, sec_orig_p)
        noise_db = cd.sec_noise_db(sec_p)
        history['iter'].append(state['count'])
        history['dyn_rmse'].append(dyn_rmse)
        history['sec_rmse'].append(sec_rmse)
        history['noise_db'].append(noise_db)
        history['pass_id'].append(state['pass'])

    print(f"[{pulsar}] Running FISTA and recording convergence metrics.")
    t_start = time.time()
    dyn_fista = fista_inpaint.fista_inpaint(
        dyn_hole, hole_mask, verbose=False, iter_callback=record)
    dt = time.time() - t_start
    n_iter = len(history['iter'])

    iters    = np.array(history['iter'])
    dyn_rmse = np.array(history['dyn_rmse'])
    sec_rmse = np.array(history['sec_rmse'])
    noise_db = np.array(history['noise_db'])
    pass_id  = np.array(history['pass_id'])

    print(
        f"[{pulsar}] Recorded {n_iter} iterations in {dt:.1f} s: "
        f"dynamic RMSE / sigma {dyn_rmse[0]:.4f} -> {dyn_rmse[-1]:.4f}, "
        f"log10 secondary RMSE / maximum {sec_rmse[0]:.4f} -> {sec_rmse[-1]:.4f}, "
        f"noise {noise_db[0]:.2f} -> {noise_db[-1]:.2f} dB.")

    # Verify the final metrics.
    rmse_dyn_final = cd.dyn_mask_rmse(dyn_fista, dyn_crop, invalid_idx)
    sec_fista_p = cd.secondary_spectrum_power(dyn_fista, hann_win)
    rmse_sec_final = cd.sec_log10_rmse(sec_fista_p, sec_orig_p)

    # Save the convergence history.
    hist_path = DATA / f'{pulsar}_iteration_history.npz'
    np.savez(hist_path, iters=iters, dyn_rmse=dyn_rmse, sec_rmse=sec_rmse,
             noise_db=noise_db, pass_id=pass_id, dyn_fista=dyn_fista)

    # Write a text summary.
    info_txt = DATA / f'{pulsar}_iteration_convergence_summary.txt'
    with open(info_txt, 'w', encoding='utf-8') as ft:
        ft.write(f"FISTA Iteration Convergence Summary — {pulsar}\n"
                 f"{'='*60}\n"
                 f"Patch: {N}×{N} at [f {f0}:{f1}, t {t0}:{t1}], hole pixels {n_hole}\n"
                 f"FISTA: default params (k=0.95), total recorded iterations = {n_iter}, "
                 f"elapsed {dt:.1f}s\n"
                 f"Initial: nRMSE_dyn={dyn_rmse[0]:.4f}, "
                 f"E_sec={sec_rmse[0]:.4f}, P_noise={noise_db[0]:.2f} dB\n"
                 f"Final:   nRMSE_dyn={dyn_rmse[-1]:.4f}, "
                 f"E_sec={sec_rmse[-1]:.4f}, P_noise={noise_db[-1]:.2f} dB\n"
                 f"Noise definition: 10*log10(mean(4 corners each N/4×N/4 of sec power)"
                 f"/DC(sec power))\n")
    print(
        f"[{pulsar}] FISTA iteration-convergence analysis complete. "
        f"Data: {hist_path}; summary: {info_txt}.")
    return {
        'iters': iters, 'dyn_rmse': dyn_rmse, 'sec_rmse': sec_rmse,
        'noise_db': noise_db, 'pass_id': pass_id,
    }

def plot_histories(results):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    fig,axes=plt.subplots(2,3,figsize=(16,8),sharex='row',layout='constrained')
    specs=[('dyn_rmse','#D95F02',r'$\mathrm{nRMSE}_{\mathrm{dyn}}$','$\mathrm{nRMSE}_{\mathrm{dyn}}$'),
           ('sec_rmse','#1B9E77',r'$E_{\mathrm{sec}}$','$E_{\mathrm{sec}}$'),
           ('noise_db','#7570B3',r'$P_{\mathrm{noise}}$ (dB)','$P_{\mathrm{noise}}$')]
    for row,psr in enumerate(cd.PULSAR_LIST):
        r=results[psr]
        for col,(key,color,ylabel,title) in enumerate(specs):
            ax=axes[row,col]
            ax.plot(r['iters'],r[key],color=color,lw=1.5)
            ax.set_xlabel('Iteration'); ax.set_ylabel(ylabel)
            ax.set_title(f'[{psr}] {title} versus iteration',fontsize=11)
            ax.text(.025,.965,f'({chr(97+3*row+col)})',transform=ax.transAxes,va='top')
            ax.grid(True,alpha=.3)
    fig.suptitle('Convergence of Dynamic- and Secondary-Spectrum Metrics during DT-FISTA Inpainting',fontsize=16)
    path=FIGURES/'fista_iteration_rmse_noise.png'
    fig.savefig(path,dpi=220,facecolor='white'); plt.close(fig)
    print(f'Saved {path}',flush=True)

if __name__ == '__main__':
    plot_histories({psr:compute_history(psr) for psr in cd.PULSAR_LIST})



