"""DT-FISTA inpainting for masked dynamic spectra.
"""

import numpy as np
from scipy.fft import fft2, ifft2
import time


# Window and edge helpers

def _edge_hanning_window(Nf, Nt, edge_frac):
    """
    Return a two-dimensional edge Hanning window.
    """
    def w1d(n):
        e = int(n * edge_frac)
        h = np.hanning(2 * e)
        w = np.ones(n)
        w[:e] = h[:e]
        w[-e:] = h[e:]
        return w

    win = np.outer(w1d(Nf), w1d(Nt))
    pos_min = np.min(win[win > 0])
    win[win <= 0] = pos_min
    return win


def _edge_band_mask(Nf, Nt, edge_frac):
    """
    Return a boolean mask for the edge band.
    """
    ef = int(Nf * edge_frac)
    et = int(Nt * edge_frac)
    band = np.zeros((Nf, Nt), dtype=bool)
    band[:ef, :] = True     # top
    band[-ef:, :] = True    # bottom
    band[:, :et] = True     # left
    band[:, -et:] = True    # right
    return band


# Core FISTA iteration

def _fista_core(d_array, mask, apply_hanning, hanning_frac,
                k, init_rank, clip_frac, threshold_floor=None,
                iter_callback=None):
    """
    Run one centered, Fourier-domain FISTA pass and return the filled array.

    iter_callback : callable, optional
        If given, called as iter_callback(it, recon) after every iteration,
        where `it` is the 1-based iteration number of this pass and `recon` is
        the current full reconstruction (un-windowed, mean restored, known
         pixels reset to their input values). Recording only; the iteration
        numerics are unchanged.
    """
    valid_idx = (mask > 0.5)
    invalid_idx = ~valid_idx
    Nf, Nt = d_array.shape

    # De-mean
    global_mean = np.mean(d_array[valid_idx])
    d_centered = d_array - global_mean

    # Optional edge Hanning window
    if apply_hanning:
        window = _edge_hanning_window(Nf, Nt, hanning_frac)
    else:
        window = np.ones((Nf, Nt))
    d_centered = d_centered * window

    # FISTA initialization
    x = d_centered.copy()
    x[invalid_idx] = 0.0      # Initialize missing samples.
    y = x.copy()              # Momentum state.
    t = 1.0

    S = fft2(y)
    mag = np.abs(S)
    thresh_val = np.partition(mag.ravel(), -init_rank)[-init_rank]   # Initial threshold.

    it = 0
    x_new = x
    def should_continue():
        if threshold_floor is not None:
            return thresh_val >= threshold_floor
        return thresh_val >= np.median(mag)

    while should_continue():
        # Soft-threshold in the FFT domain
        phase = np.angle(S)
        dc_val = S[0, 0]
        mag_thresh = np.maximum(mag - thresh_val, 0.0)
        S_new = mag_thresh * np.exp(1j * phase)
        S_new[0, 0] = dc_val
        x_rec = np.real(ifft2(S_new))

        # Enforce data consistency.
        x_new = d_centered.copy()
        x_new[invalid_idx] = x_rec[invalid_idx]

        it += 1

        # Record optional diagnostics.
        if iter_callback is not None:
            recon = x_new / window + global_mean
            recon[valid_idx] = d_array[valid_idx]
            iter_callback(it, recon)

        # FISTA momentum extrapolation
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        momentum = (t - 1.0) / t_new
        y = x_new + momentum * (x_new - x)
        y[valid_idx] = d_centered[valid_idx]   # Keep known samples fixed.

        x = x_new
        t = t_new
        thresh_val *= k  # Decrease the threshold.

        S = fft2(y)
        mag = np.abs(S)
    else:
        x = x_new

    # Undo the window and restore the mean.
    result = x / window + global_mean
    result[valid_idx] = d_array[valid_idx]     # Restore known samples.

    # Clip unstable edge values after unwindowing.
    if clip_frac is not None and clip_frac > 0:
        edge_band = _edge_band_mask(Nf, Nt, clip_frac)
        center_valid = (~edge_band) & valid_idx
        if np.any(center_valid):
            c_mean = np.mean(result[center_valid])
            c_std = np.std(result[center_valid])
            upper = c_mean + 3.0 * c_std
            result[edge_band] = np.clip(result[edge_band], 0.0, upper)

    # Restore known samples.
    out = d_array.copy()
    out[invalid_idx] = result[invalid_idx]
    return out, it


# Public API

def fista_inpaint(dyn_spec, mask,
                  hanning_edge=True, hanning_edge_frac=0.10,
                  edge_reinpaint=True, reinpaint_edge_frac=0.06,
                  k=0.95, init_rank=10,
                  verbose=True, iter_callback=None, threshold_floor=None):
    """
    Inpaint the masked region of a dynamic spectrum using FISTA.

    Parameters
    ----------
    dyn_spec : 2D ndarray
        Dynamic spectrum to inpaint. 
    mask : 2D ndarray
        Mask, 0=masked (to be filled), 1=unmasked (kept as valid).
    hanning_edge : bool, default True
        Whether to apply a Hanning window to the edges before the FFT.
    hanning_edge_frac : float, default 0.10
        Fraction of the edge taper region for the Hanning window.
    edge_reinpaint : bool, default True
        Whether to run a second FISTA pass on the edges. Specifically to fix
        edge pixels amplified by un-windowing.
    reinpaint_edge_frac : float, default 0.06
        Border width fraction covered by the second pass.
    k : float, default 0.95
        Decay factor applied to the threshold each iteration.
    init_rank : int, default 10
        The initial threshold is the init_rank-th largest current FFT magnitude
        coefficient.
    threshold_floor : float or None, default None
        If provided, continue iterating until the decaying threshold becomes
        smaller than this value. If None, retain the default median-magnitude
        stopping rule.
    verbose : bool, default True
        Whether to print one completion summary with iteration counts and timing.
    iter_callback : callable, optional
        If given, forwarded to each `_fista_core` pass and called as
        iter_callback(it, recon) after every iteration, where `it` is the
        1-based iteration number within the current pass and `recon` is the
        current full reconstruction (un-windowed, mean restored, known pixels
         reset to their input values; final edge clipping not yet applied).
        Useful for tracking error metrics during optimization.

    Returns
    -------
    filled : 2D ndarray
        The inpainted dynamic spectrum.
    """
    t_start = time.time()

    dyn_spec = np.asarray(dyn_spec, dtype=float)
    mask = np.asarray(mask)
    if dyn_spec.shape != mask.shape:
        raise ValueError(f"dyn_spec {dyn_spec.shape} and mask {mask.shape} have mismatched shapes")

    valid_idx = (mask > 0.5)
    invalid_idx = ~valid_idx

    if not np.any(invalid_idx):
        if verbose:
            print("FISTA skipped: the mask contains no missing samples.")
        return dyn_spec.copy()
    if not np.any(valid_idx):
        raise ValueError("Mask is all zeros.")

    Nf, Nt = dyn_spec.shape

    # Primary pass over the full image.
    filled, n1 = _fista_core(
        dyn_spec, mask,
        apply_hanning=hanning_edge, hanning_frac=hanning_edge_frac,
        k=k, init_rank=init_rank,
        clip_frac=hanning_edge_frac, threshold_floor=threshold_floor,
        iter_callback=iter_callback)
    n2 = 0
    refined_pixels = 0

    # Optional edge-refinement pass.
    if edge_reinpaint:
        edge_band = _edge_band_mask(Nf, Nt, reinpaint_edge_frac)
        # Refine only missing pixels inside the edge band.
        reinpaint_pixels = edge_band & invalid_idx

        if np.any(reinpaint_pixels):
            # Treat all other pixels as known.
            mask_second = np.ones((Nf, Nt), dtype=float)
            mask_second[reinpaint_pixels] = 0.0

            # Refine the first-pass result without a window.
            filled_edge, n2 = _fista_core(
                filled, mask_second,
                apply_hanning=False, hanning_frac=hanning_edge_frac,
                k=k, init_rank=init_rank,
                clip_frac=hanning_edge_frac, threshold_floor=threshold_floor,
                iter_callback=iter_callback)

            # Update only the target edge pixels.
            filled[reinpaint_pixels] = filled_edge[reinpaint_pixels]
            refined_pixels = int(reinpaint_pixels.sum())

    if verbose:
        refinement = (
            f", {n2} edge-refinement iterations over {refined_pixels} pixels"
            if refined_pixels else "")
        print(
            f"FISTA reconstruction complete: {n1} primary iterations{refinement}; "
            f"{time.time() - t_start:.2f} s.")

    return filled
