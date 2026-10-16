#!/usr/bin/env python3
"""
Shared data-loading and analysis utilities.

Analysis scripts consume the NPZ input files stored under ``data/inputs``.
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata

HERE = os.path.dirname(os.path.abspath(__file__))

# Pulsar-specific configuration
PULSARS = {
    'B1237+25': {
        'input_file': os.path.join(
            HERE, 'data', 'inputs', 'B1237+25.npz'),
        'color': 'g',
    },
    'B2021+51': {
        'input_file': os.path.join(
            HERE, 'data', 'inputs', 'B2021+51.npz'),
        'color': 'r',
    },
}
DEFAULT_PULSAR = 'B2021+51'
PULSAR_LIST = ('B1237+25', 'B2021+51')

OUTDIR = os.path.join(HERE, 'data')
FIGDIR = os.path.join(HERE, 'figures')
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)


def pulsar_outdir(pulsar):
    """Return the output directory for one pulsar's data products."""
    d = os.path.join(OUTDIR, pulsar)
    os.makedirs(d, exist_ok=True)
    return d


def combined_figure_path(filename):
    """Return the path for a publication figure containing both pulsars."""
    return os.path.join(FIGDIR, filename)


def style_subplot_grid(axes, tick_labelsize=12, panel_labelsize=15):
    """Enlarge tick labels and add row-major panel labels to a subplot grid."""
    for index, ax in enumerate(np.asarray(axes, dtype=object).ravel()):
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        ax.text(
            0.018, 0.975, f"({chr(ord('a') + index)})",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=panel_labelsize, fontweight='bold', color='black',
            bbox={'facecolor': 'white', 'edgecolor': 'none',
                  'alpha': 0.82, 'pad': 1.5},
            zorder=100)


def style_colorbar(colorbar, label_fontsize=13, tick_labelsize=12):
    """Apply publication-sized label and tick fonts to a colorbar."""
    colorbar.ax.yaxis.label.set_size(label_fontsize)
    colorbar.ax.tick_params(labelsize=tick_labelsize)


def ordered_pulsar_results(results, analysis_name):
    """Return results in ``PULSAR_LIST`` order after checking completeness."""
    missing = [pulsar for pulsar in PULSAR_LIST if pulsar not in results]
    if missing:
        missing_text = ', '.join(missing)
        raise RuntimeError(
            f"{analysis_name} is missing results for {missing_text}; "
            "the combined figure was not generated.")
    return [(pulsar, results[pulsar]) for pulsar in PULSAR_LIST]


def load_all(pulsar=DEFAULT_PULSAR, verbose=True):
    """Load one self-contained NPZ input.

    The input NPZ contains ``dyn`` and ``mask`` with shape ``(Nf, Nt)`` plus the
    complete ``time_min`` and ``freq_MHz`` coordinate axes. Masked samples in
    ``dyn`` are stored as zero and are forced to zero again on load.
    """
    if pulsar not in PULSARS:
        raise KeyError(f"Unknown pulsar {pulsar!r}; choose one of {PULSAR_LIST}.")
    npz_path = PULSARS[pulsar]['input_file']
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Input NPZ not found: {npz_path}. "
            f"Place {os.path.basename(npz_path)} in data/inputs.")

    with np.load(npz_path, allow_pickle=False) as d:
        required = {'dyn', 'mask', 'time_min', 'freq_MHz'}
        missing = sorted(required.difference(d.files))
        if missing:
            raise ValueError(f"{npz_path} is missing required arrays: {missing}")
        dyn = np.asarray(d['dyn'], dtype=float)
        mask = np.asarray(d['mask'])
        ts_min = np.asarray(d['time_min'], dtype=float)
        freqs = np.asarray(d['freq_MHz'], dtype=float)

    mask = (mask > 0.5).astype(np.uint8)
    if dyn.ndim != 2 or mask.shape != dyn.shape:
        raise ValueError(
            f"[{pulsar}] invalid NPZ shapes: dyn={dyn.shape}, mask={mask.shape}")
    if ts_min.ndim != 1 or freqs.ndim != 1:
        raise ValueError(f"[{pulsar}] time_min and freq_MHz must be one-dimensional")
    if dyn.shape != (freqs.size, ts_min.size):
        raise ValueError(
            f"[{pulsar}] axis lengths do not match dyn: "
            f"dyn={dyn.shape}, freq={freqs.size}, time={ts_min.size}")
    if not (np.isfinite(dyn).all() and np.isfinite(ts_min).all()
            and np.isfinite(freqs).all()):
        raise ValueError(f"[{pulsar}] NPZ contains non-finite values")

    dyn = dyn.copy()
    dyn[mask == 0] = 0.0

    if verbose:
        print(
            f"[{pulsar}] Loaded input dynamic spectrum: "
            f"shape={dyn.shape}, valid={np.mean(mask > 0.5)*100:.2f}%, "
            f"frequency={freqs[0]:.2f}--{freqs[-1]:.2f} MHz, "
            f"time={ts_min[0]:.2f}--{ts_min[-1]:.2f} min.")
    return dyn, mask, ts_min, freqs


def load_mask(pulsar=DEFAULT_PULSAR):
    """Load only the retained/masked-sample indicator from an input NPZ."""
    npz_path = PULSARS[pulsar]['input_file']
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Input NPZ not found: {npz_path}. "
            f"Place {os.path.basename(npz_path)} in data/inputs.")
    with np.load(npz_path, allow_pickle=False) as d:
        if 'mask' not in d.files:
            raise ValueError(f"{npz_path} does not contain a mask array")
        return (np.asarray(d['mask']) > 0.5).astype(np.uint8)


# Secondary-spectrum utilities

def make_hann_2d(Nf, Nt, edge_frac=0.1):
    """Construct a two-dimensional edge-tapered Hann window.

    Each dimension is tapered over ``edge_frac`` of its length. For a 300 x 300
    array and ``edge_frac=0.1``, 30 samples are tapered at every edge.
    """
    edge_f = max(1, int(edge_frac * Nf))
    hf = np.hanning(2 * edge_f)
    wf = np.ones(Nf)
    wf[:edge_f] = hf[:edge_f]; wf[-edge_f:] = hf[edge_f:]
    edge_t = max(1, int(edge_frac * Nt))
    ht = np.hanning(2 * edge_t)
    wt = np.ones(Nt)
    wt[:edge_t] = ht[:edge_t]; wt[-edge_t:] = ht[edge_t:]
    return np.outer(wf, wt)


def secondary_spectrum_power(dyn, window_2d):
    """Return the unnormalized, fftshifted secondary-spectrum power ``|FFT|^2``."""
    F = np.fft.fft2(dyn * window_2d)
    mag_sq = np.abs(F) ** 2
    return np.fft.fftshift(mag_sq)


def secondary_spectrum_db(dyn, window_2d):
    """Return secondary-spectrum power referenced to its DC bin in decibels.

    The fftshift operation places zero delay and zero Doppler at the center.
    """
    return power_to_db_center(secondary_spectrum_power(dyn, window_2d))


def power_to_db_center(power):
    """Convert linear power to dB relative to the zero-delay, zero-Doppler bin."""
    power = np.asarray(power, dtype=float)
    Nf, Nt = power.shape
    ref = float(power[Nf // 2, Nt // 2])
    if ref <= 0:
        return np.full(power.shape, np.nan, dtype=float)
    p = np.maximum(power, ref * 1e-300)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 10 * np.log10(p / ref)


def sec_rmse_db(power_filled, power_orig):
    """Compute full-image secondary-spectrum RMSE in the dB domain.

    Each spectrum is first referenced to its own zero-delay, zero-Doppler bin.
    """
    db_f = power_to_db_center(power_filled)
    db_o = power_to_db_center(power_orig)
    diff = (db_f - db_o).ravel()
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(diff ** 2)))


def sec_log10_rmse(power_filled, power_orig):
    """Compute log10 normalized RMSE in the linear-power domain.

    The RMSE is normalized by the DC power of the original secondary spectrum:
    ``sqrt(mean((filled - original)**2)) / original[DC]``.
    """
    diff = (np.asarray(power_filled, dtype=float)
            - np.asarray(power_orig, dtype=float)).ravel()
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    power_orig = np.asarray(power_orig, dtype=float)
    Nf, Nt = power_orig.shape
    ref = float(power_orig[Nf // 2, Nt // 2])
    if rmse <= 0 or ref <= 0:
        return np.nan
    return float(np.log10(rmse / ref))


def sec_noise_db(power):
    """Estimate the secondary-spectrum noise level from its four corners.

    Each corner spans one quarter of each array dimension. The mean linear
    power in the combined corner pixels is normalized by the DC power and
    converted to dB.
    """
    p = np.asarray(power, dtype=float)
    Nf, Nt = p.shape
    cf, ct = max(1, Nf // 4), max(1, Nt // 4)
    corners = np.concatenate([
        p[:cf, :ct].ravel(),  p[:cf, -ct:].ravel(),
        p[-cf:, :ct].ravel(), p[-cf:, -ct:].ravel()])
    m  = float(np.mean(corners))
    dc = float(p[Nf // 2, Nt // 2])
    if m <= 0 or dc <= 0:
        return np.nan
    return float(10.0 * np.log10(m / dc))


def dyn_mask_rmse(filled, orig, invalid_idx):
    """Compute normalized RMSE over masked dynamic-spectrum samples.

    The denominator is the standard deviation of the full original patch.
    """
    rmse = float(np.sqrt(np.mean((filled[invalid_idx] - orig[invalid_idx]) ** 2)))
    sigma = float(np.std(orig))
    if sigma > 0:
        return rmse / sigma
    return rmse


def gini_coefficient(x):
    """Compute the Gini coefficient as a sparsity measure.

    A value of 0 denotes a uniform array, while values approaching 1 indicate
    increasing sparsity. All pixels are used at full resolution.
    """
    v = np.abs(np.asarray(x, dtype=float)).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    v.sort()
    n = v.size
    s = v.sum()
    if s <= 0:
        return np.nan
    i = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(i * v)) / (n * s) - (n + 1.0) / n)


def collect_clean_patches_nonoverlap(mask, N, n_collect, seed=0,
                                     max_overlap_frac=0.0, max_masked_frac=0.0):
    """Collect patches subject to masked-area and overlap constraints.

    Candidate ``N x N`` patches may contain at most ``max_masked_frac * N^2``
    masked pixels. Pairwise overlap is limited to ``max_overlap_frac * N^2``.
    An integral image identifies candidates, and a seeded greedy selection with
    a spatial index enforces the overlap constraint. Returns integer arrays of
    frequency and time start indices.
    """
    Nf, Nt = mask.shape
    valid = (mask > 0.5).astype(np.int32)
    S = np.zeros((Nf + 1, Nt + 1), dtype=np.int64)
    S[1:, 1:] = valid.cumsum(axis=0).cumsum(axis=1)
    winsum = (S[N:, N:] - S[:-N, N:] - S[N:, :-N] + S[:-N, :-N])
    masked_max = int(max_masked_frac * N * N)
    # A zero threshold restricts candidates to completely unmasked patches.
    f0s_all, t0s_all = np.where(winsum >= N * N - masked_max)
    n_cand = len(f0s_all)
    if n_cand == 0:
        raise RuntimeError(
            f"No {N} x {N} patch has a masked fraction at or below "
            f"{max_masked_frac*100:.1f}%.")
    if n_collect is None:
        n_collect = n_cand

    # Seeded random priorities make the greedy selection reproducible.
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_cand)
    max_ov = max_overlap_frac * N * N

    # Index selected patches in N-pixel cells to limit overlap comparisons.
    grid = {}
    sel_f, sel_t = [], []
    for idx in order:
        f0, t0 = int(f0s_all[idx]), int(t0s_all[idx])
        f1, t1 = f0 + N, t0 + N
        cf0, ct0 = f0 // N, t0 // N
        cf1, ct1 = (f1 - 1) // N, (t1 - 1) // N
        # Exact overlap checks are needed only for candidates in nearby cells.
        rejected = False
        for gf in range(cf0, cf1 + 1):
            if rejected:
                break
            for gt in range(ct0, ct1 + 1):
                for sf, st in grid.get((gf, gt), ()):
                    of = min(f1, sf + N) - max(f0, sf)
                    if of <= 0:
                        continue
                    ot = min(t1, st + N) - max(t0, st)
                    if ot <= 0:
                        continue
                    if of * ot > max_ov:
                        rejected = True
                        break
                if rejected:
                    break
        if rejected:
            continue
        sel_f.append(f0)
        sel_t.append(t0)
        for gf in range(cf0, cf1 + 1):
            for gt in range(ct0, ct1 + 1):
                grid.setdefault((gf, gt), []).append((f0, t0))
        if len(sel_f) >= n_collect:
            break

    n_sel = len(sel_f)
    print(
        f"Selected {n_sel} of {n_cand} candidate {N} x {N} patches "
        f"(masked <= {max_masked_frac*100:.1f}%, pairwise overlap <= "
        f"{max_overlap_frac*100:.0f}%, seed={seed}).")
    return np.array(sel_f, dtype=int), np.array(sel_t, dtype=int)


def collect_clean_patches(mask, N, n_collect, seed=0):
    """Collect reproducibly sampled, fully unmasked ``N x N`` patches.

    An integral image finds all valid locations. Sampling is without replacement;
    all available patches are returned when fewer than ``n_collect`` exist.
    """
    Nf, Nt = mask.shape
    valid = (mask > 0.5).astype(np.int32)
    S = np.zeros((Nf + 1, Nt + 1), dtype=np.int64)
    S[1:, 1:] = valid.cumsum(axis=0).cumsum(axis=1)
    winsum = (S[N:, N:] - S[:-N, N:] - S[N:, :-N] + S[:-N, :-N])
    f0s_all, t0s_all = np.where(winsum == N * N)
    n_cand = len(f0s_all)
    if n_cand == 0:
        raise RuntimeError(f"No fully valid {N} x {N} patch was found.")
    n_sel = min(n_collect, n_cand)
    rng = np.random.RandomState(seed)
    sel = rng.choice(n_cand, size=n_sel, replace=False)
    sel.sort()
    print(
        f"Selected {n_sel} of {n_cand} fully valid {N} x {N} patches "
        f"(seed={seed}).")
    return f0s_all[sel].astype(int), t0s_all[sel].astype(int)


def make_sec_edges(Nf, Nt, freqs_MHz, ts_s):
    """Construct secondary-spectrum bin edges for plotting.

    Returns delay edges in microseconds and Doppler edges in millihertz.
    """
    df_MHz = abs(np.median(np.diff(freqs_MHz)))
    dt_s   = abs(np.median(np.diff(ts_s)))
    ft_ext = np.fft.fftshift(np.fft.fftfreq(Nf, d=df_MHz))
    half_bin_delay = 0.5 / (Nf * df_MHz)
    ft_edges = np.linspace(ft_ext[0] - half_bin_delay, ft_ext[-1] + half_bin_delay, Nf + 1)
    t_ext = np.fft.fftshift(np.fft.fftfreq(Nt, d=dt_s)) * 1000.0
    half_bin_fD = 0.5 * 1000.0 / (Nt * dt_s)
    t_edges = np.linspace(t_ext[0] - half_bin_fD, t_ext[-1] + half_bin_fD, Nt + 1)
    return ft_edges, t_edges


# RFI interpolation

def fill_rfi_interp(dyn_patch, rfi_mask):
    """Fill RFI-masked samples by two-dimensional linear interpolation.

    Nearest-neighbor interpolation handles samples outside the convex hull.
    ``rfi_mask`` is true where values must be filled.
    """
    valid = ~rfi_mask
    f_valid, t_valid = np.where(valid)
    values = dyn_patch[f_valid, t_valid]
    f_rfi, t_rfi = np.where(rfi_mask)
    filled = griddata((f_valid, t_valid), values, (f_rfi, t_rfi), method='linear')
    nan_mask = np.isnan(filled)
    if np.any(nan_mask):
        filled_nn = griddata((f_valid, t_valid), values,
                             (f_rfi[nan_mask], t_rfi[nan_mask]), method='nearest')
        filled[nan_mask] = filled_nn
    result = dyn_patch.copy()
    result[f_rfi, t_rfi] = filled
    return result


# Synthetic-gap generation

def _exact_hole_mask_from_scores(scores, hole_area):
    """Select exactly ``hole_area`` pixels with the smallest finite scores."""
    scores = np.asarray(scores, dtype=float)
    target = int(hole_area)
    if target != hole_area or target <= 0:
        raise ValueError("hole_area must be a positive integer.")

    flat_scores = scores.ravel()
    candidates = np.flatnonzero(np.isfinite(flat_scores))
    if target > candidates.size:
        raise ValueError(
            f"Requested {target} hole pixels but only {candidates.size} "
            "candidate pixels are available.")

    # The flat index provides a deterministic tie break at pixelized boundaries.
    order = np.lexsort((candidates, flat_scores[candidates]))
    selected = candidates[order[:target]]
    mask = np.ones(scores.shape, dtype=bool)
    mask.ravel()[selected] = False
    return mask


def make_circle_hole(N, hole_area):
    """Create a pixelized circular gap with exactly ``hole_area`` pixels."""
    f_center = 3 * N // 4
    t_center = 3 * N // 4
    ff, tt = np.indices((N, N))
    distance_sq = (ff - f_center)**2 + (tt - t_center)**2
    mask = _exact_hole_mask_from_scores(distance_sq, hole_area)
    r = np.sqrt(hole_area / np.pi)
    actual_area = int(np.sum(~mask))
    return mask, (f_center, t_center, r, actual_area)


def make_irregular_hole(N, hole_area, random_seed=0):
    """Create a diagonal irregular gap running from upper right to lower left."""
    rng = np.random.RandomState(random_seed)
    t_left  = int(0.10 * N)
    t_right = int(0.90 * N)
    width   = t_right - t_left
    avg_h   = hole_area / width
    half_h  = avg_h / 2.0
    f_center_line = N - np.arange(t_left, t_right)
    t = np.arange(width)
    n_modes = 5
    pert_top = np.zeros(width)
    pert_bot = np.zeros(width)
    for k in range(1, n_modes + 1):
        pert_top += rng.uniform(0.4, 1.0) * np.sin(
            2 * np.pi * k * t / width + rng.uniform(0, 2 * np.pi))
        pert_bot += rng.uniform(0.4, 1.0) * np.cos(
            2 * np.pi * k * t / width + rng.uniform(0, 2 * np.pi))
    pert_top = gaussian_filter1d(pert_top, sigma=4.0)
    pert_bot = gaussian_filter1d(pert_bot, sigma=4.0)
    pert_amp = 0.5 * half_h
    if np.std(pert_top) > 1e-10:
        pert_top = pert_top * pert_amp / np.std(pert_top)
    if np.std(pert_bot) > 1e-10:
        pert_bot = pert_bot * pert_amp / np.std(pert_bot)
    top_b = np.maximum(half_h + pert_top, 1.0)
    bot_b = np.maximum(half_h + pert_bot, 1.0)
    scores = np.full((N, N), np.inf, dtype=float)
    frequency_indices = np.arange(N)
    for i, t_idx in enumerate(range(t_left, t_right)):
        fc = f_center_line[i]
        offsets = frequency_indices - fc
        local_scale = np.where(offsets >= 0, top_b[i], bot_b[i])
        scores[:, t_idx] = np.abs(offsets) / local_scale

    hole_mask = _exact_hole_mask_from_scores(scores, hole_area)
    actual_area = int(np.sum(~hole_mask))

    # Store the realized boundaries so the plotted outline follows the mask.
    realized_top = np.zeros(width, dtype=float)
    realized_bottom = np.zeros(width, dtype=float)
    for i, t_idx in enumerate(range(t_left, t_right)):
        hole_rows = np.flatnonzero(~hole_mask[:, t_idx])
        if hole_rows.size:
            fc = f_center_line[i]
            realized_top[i] = max(0, hole_rows[-1] + 1 - fc)
            realized_bottom[i] = max(0, fc - hole_rows[0])
    return hole_mask, (f_center_line, t_left, t_right, avg_h, actual_area,
                        realized_top, realized_bottom, width)


def make_square_hole(N, hole_area):
    """Create a pixelized square gap with exactly ``hole_area`` pixels."""
    f_center = N // 4
    t_center = N // 4
    ff, tt = np.indices((N, N))
    df = np.abs(ff - f_center)
    dt = np.abs(tt - t_center)
    chebyshev_distance = np.maximum(df, dt)
    # Spread remainder pixels around the square boundary.
    square_scores = (chebyshev_distance
                     + (df**2 + dt**2) / float(10 * N * N))
    mask = _exact_hole_mask_from_scores(square_scores, hole_area)
    hole_f, hole_t = np.where(~mask)
    f1, f2 = int(hole_f.min()), int(hole_f.max()) + 1
    t1, t2 = int(hole_t.min()), int(hole_t.max()) + 1
    s = np.sqrt(hole_area)
    actual_area = int(np.sum(~mask))
    return mask, (f1, f2, t1, t2, f_center, t_center, s, actual_area)


def make_center_square_hole(N, hole_area):
    """Create a square gap at the patch center for the size-scan experiment."""
    s = int(np.sqrt(hole_area))
    f_center = N // 2
    t_center = N // 2
    f1 = f_center - s // 2
    t1 = t_center - s // 2
    f2 = f1 + s
    t2 = t1 + s
    mask = np.ones((N, N), dtype=bool)
    mask[f1:f2, t1:t2] = False
    actual_area = np.sum(~mask)
    return mask, (f1, f2, t1, t2, f_center, t_center, s, actual_area)


def make_combined_hole(N, hole_area, random_seed=0):
    """Combine circular, irregular, and square synthetic gaps."""
    circle_mask, circle_info    = make_circle_hole(N, hole_area)
    irregular_mask, irreg_info  = make_irregular_hole(N, hole_area, random_seed)
    square_mask, square_info    = make_square_hole(N, hole_area)
    combined = circle_mask & irregular_mask & square_mask
    total_hole_pixels = int(np.sum(~combined))
    info = {
        'circle':    circle_info,
        'irregular': irreg_info,
        'square':    square_info,
        'total_hole_pixels': total_hole_pixels,
    }
    return combined, info


# Clean-patch search

def find_clean_patch_nearest_center(mask, N):
    """Find the fully unmasked ``N x N`` patch nearest the array center.

    An integral image accelerates the search. Returns the frequency and time
    indices of the patch's lower-left array corner. Mask values are 1 for valid
    samples and 0 for RFI.
    """
    Nf, Nt = mask.shape
    valid = (mask > 0.5).astype(np.int32)
    # Evaluate all candidate windows with the integral image.
    S = np.zeros((Nf + 1, Nt + 1), dtype=np.int64)
    S[1:, 1:] = valid.cumsum(axis=0).cumsum(axis=1)
    # Sum every N x N window at its candidate start index.
    winsum = (S[N:, N:] - S[:-N, N:] - S[N:, :-N] + S[:-N, :-N])
    clean = (winsum == N * N)
    n_clean = int(np.sum(clean))
    if n_clean == 0:
        raise RuntimeError(f"No fully valid {N} x {N} patch was found.")
    f0s, t0s = np.where(clean)
    # Rank candidates by distance from the array center.
    fc = f0s + (N - 1) / 2.0 - (Nf - 1) / 2.0
    tc = t0s + (N - 1) / 2.0 - (Nt - 1) / 2.0
    dist2 = fc ** 2 + tc ** 2
    idx = int(np.argmin(dist2))
    f0, t0 = int(f0s[idx]), int(t0s[idx])
    print(
        f"Selected the most central of {n_clean} fully valid {N} x {N} patches: "
        f"[f {f0}:{f0+N}, t {t0}:{t0+N}], offset "
        f"(df={fc[idx]:.1f}, dt={tc[idx]:.1f}) pixels.")
    return f0, t0, n_clean


def find_clean_patch_random(mask, N, seed=0, max_masked_frac=0.0):
    """Select a reproducible random patch below a masked-area threshold.

    ``max_masked_frac=0`` requires a fully clean patch. Returns the selected
    frequency index, time index, and total number of candidates.
    """
    Nf, Nt = mask.shape
    valid = (mask > 0.5).astype(np.int32)
    S = np.zeros((Nf + 1, Nt + 1), dtype=np.int64)
    S[1:, 1:] = valid.cumsum(axis=0).cumsum(axis=1)
    winsum = (S[N:, N:] - S[:-N, N:] - S[N:, :-N] + S[:-N, :-N])
    masked_max = int(max_masked_frac * N * N)
    f0s, t0s = np.where(winsum >= N * N - masked_max)
    n_cand = len(f0s)
    if n_cand == 0:
        raise RuntimeError(
            f"No {N} x {N} patch has a masked fraction at or below "
            f"{max_masked_frac*100:.1f}%.")
    rng = np.random.RandomState(seed)
    idx = int(rng.randint(n_cand))
    f0, t0 = int(f0s[idx]), int(t0s[idx])
    condition = ("fully valid" if masked_max == 0
                 else f"masked <= {max_masked_frac*100:.1f}%")
    print(
        f"Selected candidate {idx + 1} of {n_cand} {N} x {N} patches "
        f"({condition}, seed={seed}): [f {f0}:{f0+N}, t {t0}:{t0+N}].")
    return f0, t0, n_cand


def find_clean_patch_random_in_band(mask, N, freqs_MHz, f_min_MHz, f_max_MHz, seed=0):
    """Select a reproducible clean patch fully contained in a frequency band.

    Returns the selected frequency index, time index, and number of candidates.
    """
    Nf, Nt = mask.shape
    valid = (mask > 0.5).astype(np.int32)
    S = np.zeros((Nf + 1, Nt + 1), dtype=np.int64)
    S[1:, 1:] = valid.cumsum(axis=0).cumsum(axis=1)
    winsum = (S[N:, N:] - S[:-N, N:] - S[N:, :-N] + S[:-N, :-N])
    clean = (winsum == N * N)

    # Enforce the requested frequency band.
    f0_lo = int(np.searchsorted(freqs_MHz, f_min_MHz, side='left'))
    f0_hi = int(np.searchsorted(freqs_MHz, f_max_MHz, side='right')) - N
    if f0_hi < f0_lo:
        raise RuntimeError(
            f"The {f_min_MHz}--{f_max_MHz} MHz band cannot contain a patch "
            f"that spans {N} channels.")
    band_clean = np.zeros_like(clean)
    band_clean[f0_lo:f0_hi + 1, :] = clean[f0_lo:f0_hi + 1, :]

    f0s, t0s = np.where(band_clean)
    n_cand = len(f0s)
    if n_cand == 0:
        raise RuntimeError(
            f"No fully valid {N} x {N} patch was found in the "
            f"{f_min_MHz}--{f_max_MHz} MHz band.")
    rng = np.random.RandomState(seed)
    idx = int(rng.randint(n_cand))
    f0, t0 = int(f0s[idx]), int(t0s[idx])
    print(
        f"Selected candidate {idx + 1} of {n_cand} fully valid {N} x {N} "
        f"patches in the {f_min_MHz}--{f_max_MHz} MHz band (seed={seed}): "
        f"[f {f0}:{f0+N}, t {t0}:{t0+N}], "
        f"{freqs_MHz[f0]:.2f}--{freqs_MHz[f0+N-1]:.2f} MHz.")
    return f0, t0, n_cand


# Synthetic-gap boundary overlays

def draw_hole_markers(ax, hole_info, ts_arr, freqs_arr, N, color='black'):
    """Draw the three synthetic-gap boundaries on an axis."""
    import matplotlib.patches as mpatches
    dt = abs(np.median(np.diff(ts_arr)))
    df = abs(np.median(np.diff(freqs_arr)))

    # Gap 1: square in the upper-left part of the patch.
    si = hole_info['square']
    f1_s, f2_s, t1_s, t2_s = si[0], si[1], si[2], si[3]
    rect = mpatches.Rectangle(
        (ts_arr[t1_s], freqs_arr[f1_s]),
        ts_arr[t2_s - 1] - ts_arr[t1_s],
        freqs_arr[f2_s - 1] - freqs_arr[f1_s],
        fill=False, edgecolor=color, linewidth=1.5, linestyle='-', zorder=10)
    ax.add_patch(rect)

    # Gap 2: irregular diagonal strip.
    ii = hole_info['irregular']
    t_left, t_right = ii[1], ii[2]
    top_b, bot_b = ii[5], ii[6]
    f_center_line = ii[0]
    t_vals = ts_arr[t_left:t_right]
    f_top_vals = np.array([freqs_arr[min(N - 1, f_center_line[i] + int(np.round(top_b[i])))]
                           for i in range(len(t_vals))])
    f_bot_vals = np.array([freqs_arr[max(0, f_center_line[i] - int(np.round(bot_b[i])))]
                           for i in range(len(t_vals))])
    ax.plot(t_vals, f_top_vals, color=color, linewidth=1.2, zorder=10)
    ax.plot(t_vals, f_bot_vals, color=color, linewidth=1.2, zorder=10)
    ax.plot([ts_arr[t_left], ts_arr[t_left]],
            [f_bot_vals[0], f_top_vals[0]],
            color=color, linewidth=1.2, zorder=10)
    ax.plot([ts_arr[t_right - 1], ts_arr[t_right - 1]],
            [f_bot_vals[-1], f_top_vals[-1]],
            color=color, linewidth=1.2, zorder=10)

    # Gap 3: circle in the lower-right region.
    ci = hole_info['circle']
    f_ctr_c, t_ctr_c, r = ci[0], ci[1], ci[2]
    ax.add_patch(mpatches.Ellipse(
        (ts_arr[t_ctr_c], freqs_arr[f_ctr_c]),
        width=2 * r * dt, height=2 * r * df,
        fill=False, edgecolor=color, linewidth=1.5, linestyle='-', zorder=10))



