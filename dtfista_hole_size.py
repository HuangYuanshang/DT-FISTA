"""Measure DT-FISTA performance as a central square mask grows.

Run from this directory with ``python dtfista_hole_size.py``.
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

HOLE_FRACS = [.04, .08, .16, .32, .64]

def make_center_square_hole(N, hole_area):
    area=int(hole_area)
    s=int(np.sqrt(area))
    f1=t1=N//2-s//2
    f2,t2=f1+s,t1+s
    assert 0<f1<f2<N and 0<t1<t2<N
    missing=np.zeros((N,N),dtype=bool)
    missing[f1:f2,t1:t2]=True
    remainder=area-s*s
    # Add the remainder along a connected boundary path.
    perimeter=([(f1+i,t2) for i in range(s)]+
               [(f2,t2-i) for i in range(s+1)])
    if remainder:
        for pixel in perimeter[:remainder]:
            missing[pixel]=True
    assert int(missing.sum())==area
    # Return geometry metadata for plotting.
    return ~missing, (f1,f2,t1,t2,N//2,N//2,s,area)

def compute_scan(pulsar='B2021+51'):
    print(f"[{pulsar}] Starting the central-gap size sensitivity analysis.")

    dyn_crop, ts_crop, freqs_crop, crop = reference_patch(pulsar)
    f0, f1, t0, t1 = crop

    hann_win = cd.make_hann_2d(N, N)
    sec_orig_p = cd.secondary_spectrum_power(dyn_crop, hann_win)

    # Evaluate each requested mask fraction.
    results = {'frac': [], 'side': [], 'n_hole': [],
               'dyn_rmse': [], 'sec_rmse': [], 'n_iter_equiv_time': [],
               'dyn_rmse_mean': [], 'sec_rmse_mean': [], 'noise_db': []}
    filled_64 = None
    hole_mask_64 = None
    dyn_hole_64 = None

    for frac in HOLE_FRACS:
        hole_area = round(frac * N * N)
        hole_mask_combined, info = make_center_square_hole(N, hole_area)
        hole_mask_int = hole_mask_combined.astype(np.int8)
        invalid_idx = (hole_mask_int < 0.5)
        n_hole = int(np.sum(invalid_idx))
        assert n_hole == hole_area
        actual_frac = n_hole / (N * N)
        s = info[6]

        dyn_hole = dyn_crop.copy()
        dyn_hole[~hole_mask_combined] = np.mean(dyn_crop[hole_mask_int > 0.5])

        t_start = time.time()
        dyn_fista = fista_inpaint.fista_inpaint(
            dyn_hole, hole_mask_int, verbose=False)
        dt = time.time() - t_start

        rmse_dyn = cd.dyn_mask_rmse(dyn_fista, dyn_crop, invalid_idx)
        sec_p = cd.secondary_spectrum_power(dyn_fista, hann_win)
        rmse_sec = cd.sec_log10_rmse(sec_p, sec_orig_p)

        # Evaluate the mean-filled baseline.
        rmse_dyn_mean = cd.dyn_mask_rmse(dyn_hole, dyn_crop, invalid_idx)
        sec_hole_p = cd.secondary_spectrum_power(dyn_hole, hann_win)
        rmse_sec_mean = cd.sec_log10_rmse(sec_hole_p, sec_orig_p)

        # Estimate corner-background noise.
        noise_db = cd.sec_noise_db(sec_p)

        results['frac'].append(frac)
        results['side'].append(s)
        results['n_hole'].append(n_hole)
        results['dyn_rmse'].append(rmse_dyn)
        results['sec_rmse'].append(rmse_sec)
        results['dyn_rmse_mean'].append(rmse_dyn_mean)
        results['sec_rmse_mean'].append(rmse_sec_mean)
        results['noise_db'].append(noise_db)
        results['n_iter_equiv_time'].append(dt)
        print(
            f"[{pulsar}] Missing area {actual_frac*100:.2f}% "
            f"({n_hole} pixels): dynamic RMSE / sigma={rmse_dyn:.4f} "
            f"(mean fill {rmse_dyn_mean:.4f}), log10 secondary RMSE / DC="
            f"{rmse_sec:.4f} (mean fill {rmse_sec_mean:.4f}), "
            f"noise={noise_db:.2f} dB, elapsed={dt:.1f} s.")

        if abs(frac - 0.64) < 1e-9:
            filled_64 = dyn_fista
            hole_mask_64 = hole_mask_int
            dyn_hole_64 = dyn_hole.copy()

    fracs        = np.array(results['frac'])
    dyn_rmse     = np.array(results['dyn_rmse'])
    sec_rmse     = np.array(results['sec_rmse'])
    dyn_rmse_mean = np.array(results['dyn_rmse_mean'])
    sec_rmse_mean = np.array(results['sec_rmse_mean'])
    noise_db_arr  = np.array(results['noise_db'])

    # Save scan results and the 64% reconstruction.
    npz_out = DATA / f'{pulsar}_hole_size_scan.npz'
    np.savez(npz_out, fracs=fracs, sides=np.array(results['side']),
             n_hole=np.array(results['n_hole']),
             dyn_rmse=dyn_rmse, sec_rmse=sec_rmse,
             dyn_rmse_mean=dyn_rmse_mean, sec_rmse_mean=sec_rmse_mean,
             noise_db=noise_db_arr,
             dyn_fista_64=filled_64, hole_mask_64=hole_mask_64,
             dyn_hole_64=dyn_hole_64,
             sec_rmse_unit='log10(sqrt(mean((sec_filled-sec_orig)^2))/DC(sec_orig)), linear power, no dB',
             dyn_rmse_unit='sqrt(mean((filled[hole]-orig[hole])^2))/std(orig)',
             noise_unit='10*log10(mean(4 corners each N/4×N/4 of sec power)/DC(sec power)) [dB]')

    # Write a text summary.
    info_txt = DATA / f'{pulsar}_hole_size_sensitivity_summary.txt'
    noise_orig = cd.sec_noise_db(sec_orig_p)
    with open(info_txt, 'w', encoding='utf-8') as ft:
        ft.write(f"Missing-Area Sensitivity Summary — {pulsar}\n"
                 f"{'='*60}\n"
                 f"Patch: {N}×{N} at [f {f0}:{f1}, t {t0}:{t1}]\n"
                 f"Central square holes: {', '.join(f'{int(f*100)}%' for f in HOLE_FRACS)}, "
                 f"DT-FISTA default params (k=0.95)\n"
                 f"Original spectrum noise = {noise_orig:.2f} dB\n\n")
        ft.write(f"{'hole%':>6} {'side':>5} {'nRMSE_dyn':>11} "
                 f"{'E_sec':>19} {'P_noise(dB)':>10} {'time(s)':>8}\n")
        for i, frac in enumerate(fracs):
            ft.write(f"{frac*100:6.0f} {results['side'][i]:5d} "
                     f"{dyn_rmse[i]:11.4f} {sec_rmse[i]:19.4f} "
                     f"{noise_db_arr[i]:10.2f} {results['n_iter_equiv_time'][i]:8.1f}\n")
        ft.write(f"\nMean-Filled baseline (for reference):\n")
        for i, frac in enumerate(fracs):
            ft.write(f"  hole {frac*100:3.0f}%: nRMSE_dyn={dyn_rmse_mean[i]:.4f}, "
                     f"E_sec={sec_rmse_mean[i]:.4f}\n")

    # Prepare the 64% comparison spectra.
    sec_fista64_p = cd.secondary_spectrum_power(filled_64, hann_win)
    sec_hole64_p  = cd.secondary_spectrum_power(dyn_hole_64, hann_win)
    sec_orig_db   = cd.power_to_db_center(sec_orig_p)
    sec_hole64_db = cd.power_to_db_center(sec_hole64_p)
    sec_fista64_db = cd.power_to_db_center(sec_fista64_p)

    print(
        f"[{pulsar}] Central-gap size sensitivity analysis complete. "
        f"Data: {npz_out}; summary: {info_txt}.")
    return {
        'fracs': fracs, 'dyn_rmse': dyn_rmse, 'sec_rmse': sec_rmse,
        'dyn_rmse_mean': dyn_rmse_mean, 'sec_rmse_mean': sec_rmse_mean,
        'noise_db': noise_db_arr, 'noise_orig': noise_orig,
        'dyn_crop': dyn_crop, 'dyn_hole_64': dyn_hole_64,
        'filled_64': filled_64, 'sec_orig_db': sec_orig_db,
        'sec_hole64_db': sec_hole64_db, 'sec_fista64_db': sec_fista64_db,
        'ts_crop': ts_crop, 'freqs_crop': freqs_crop, 'N': N,
    }

def plot_combined(results):
    """Combine the size trends and 64% DT-FISTA images into two rows."""
    plt.rcParams.update({'font.size':13,'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11})
    fig,axes=plt.subplots(2,4,figsize=(18,8.7),layout='constrained')
    fig.set_layout_engine('constrained',w_pad=.03,h_pad=.04,wspace=.02,hspace=.06,rect=(0,0,1,.945))
    fig.suptitle('Inpainting error versus masked area',fontsize=18,y=.99)
    for row,psr in enumerate(cd.PULSAR_LIST):
        r=results[psr]
        for col,key in enumerate(('dyn_rmse','sec_rmse')):
            ax=axes[row,col]
            ax.plot(r['fracs']*100,r[key],'o-',color='black',lw=1.4,ms=4.5)
            ax.set_xticks([4,8,16,32,64]);ax.set_xlabel('Masked area (%)')
            ax.set_ylabel(r'$\mathrm{nRMSE}_{\mathrm{dyn}}$' if col==0 else
                r'$E_{\mathrm{sec}}$')
            metric = r'$\mathrm{nRMSE}_{\mathrm{dyn}}$' if col==0 else r'$E_{\mathrm{sec}}$'
            ax.set_title(f'[{psr}] {metric}\nversus masked area',fontsize=13)
            ax.grid(True,color='.88',lw=.6)
        ts,fs=r['ts_crop'],r['freqs_crop']
        delay,doppler=cd.make_sec_edges(r['N'],r['N'],fs,ts*60)
        secmin,secmax=np.percentile(r['sec_orig_db'],[50,99.9]);secmin-=5
        dynim=axes[row,2].pcolormesh(ts,fs,r['filled_64'],shading='auto',cmap='viridis',
            norm=matplotlib.colors.Normalize(0,np.percentile(r['dyn_crop'],99.5)))
        secim=axes[row,3].pcolormesh(doppler,delay,r['sec_fista64_db'],shading='auto',cmap='inferno',
            norm=matplotlib.colors.Normalize(secmin,secmax),rasterized=True)
        axes[row,2].collections[0].set_rasterized(True)
        axes[row,2].set_xlabel('Time (min)');axes[row,2].set_ylabel('Frequency (MHz)')
        axes[row,3].set_xlabel('Doppler (mHz)');axes[row,3].set_ylabel(r'Delay ($\mu$s)')
        axes[row,2].set_title(f'[{psr}] Dynamic spectrum\nDT-FISTA; 64% mask region',fontsize=13)
        axes[row,3].set_title(f'[{psr}] Secondary spectrum\nDT-FISTA; 64% mask region',fontsize=13)
        for col,im,title in [(2,dynim,'Intensity'),(3,secim,'dB')]:
            cb=fig.colorbar(im,ax=axes[row,col],location='right',fraction=.046,pad=.04,shrink=.85)
            cb.set_label(title,fontsize=12);cb.ax.tick_params(labelsize=11)
        for col in range(4):
            ax=axes[row,col];ax.set_box_aspect(1);ax.tick_params(labelsize=11)
            ax.text(.025,.965,f'({chr(97+4*row+col)})',transform=ax.transAxes,va='top',fontsize=13,
                color='white' if col>1 else 'black')
    plt.rcParams['pdf.fonttype']=42
    fig.savefig(OUT/'dtfista_hole_size.png',dpi=240,facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    plot_combined({psr:compute_scan(psr) for psr in cd.PULSAR_LIST})
    print(f"Saved {FIGURES / 'dtfista_hole_size.png'}",flush=True)



