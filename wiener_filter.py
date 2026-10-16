"""Dense Wiener reconstruction for masked dynamic spectra.

Implements the covariance-based method described by Lin et al. (2021) https://doi.org/10.1093/mnras/stab1811.

Xun Shi
"""

import gc
import time

import numpy as np
from scipy.sparse.linalg import cg


def secondary_spectrum_power(dyn, npad=0):
    """Return fftshifted secondary-spectrum power for a dynamic spectrum.

    ``npad`` appends that many array lengths of zero-padding to the positive end
    of each dimension. The default preserves the input shape.
    """
    dyn = np.asarray(dyn)
    padded = np.pad(
        dyn,
        ((0, dyn.shape[0] * npad), (0, dyn.shape[1] * npad)),
    )
    spectrum = np.fft.fftn(padded)
    return np.abs(np.fft.fftshift(spectrum)) ** 2


def wiener_filter(d_array_binned, H, NAME_KEY='', verbose=True):
    """Fill masked samples with a Wiener reconstruction.

    ``H`` uses 1 for known samples and 0 for missing samples. Known values are
    retained in the covariance solve. When ``NAME_KEY`` is non-empty, the full
    reconstruction is also saved as ``s_WF_<NAME_KEY>.dat``. Set ``verbose``
    to false to suppress routine status messages.
    """

    if np.where(H==0.0)[0].size == 0:
        if verbose:
            print("Wiener reconstruction skipped: the mask contains no missing samples.")
        return d_array_binned
    else:
        start = time.time()
        if verbose:
            print(f"Running Wiener reconstruction for an array with shape {d_array_binned.shape}.")

        sec = secondary_spectrum_power(d_array_binned)

        autocorr_fft = np.fft.ifftn(np.fft.ifftshift(sec)).real
        
        autocorr_fft_max = autocorr_fft.max()
        autocorr_fft = autocorr_fft / autocorr_fft_max

        # Estimate the noise from four corner regions, each spanning 1/4 x 1/4.
        Nf_sec, Nt_sec = sec.shape
        ch, cw = Nf_sec // 4, Nt_sec // 4
        corners_mean = np.mean([
            np.mean(sec[:ch, :cw]),
            np.mean(sec[:ch, -cw:]),
            np.mean(sec[-ch:, :cw]),
            np.mean(sec[-ch:, -cw:]),
        ])
        var_n = corners_mean / autocorr_fft_max

        def iy1ix1_2_i(iy1, ix1):
            """Map two-dimensional indices to a flat index."""
            return d_array_binned.shape[1] * iy1 + ix1

        def i_2_iy1ix1_2_i(i):
            """Map a flat index to two-dimensional indices."""
            iy1 = i // d_array_binned.shape[1] 
            ix1 = i - d_array_binned.shape[1] * iy1
            return iy1, ix1

        def masked_j():
            """Return flat indices of missing samples."""
            iy2_masked, ix2_masked = np.where(H==0.)
            j_masked = [iy1ix1_2_i(iy2_masked[n], ix2_masked[n]) for n in range(iy2_masked.size)]
            return j_masked

        HSHpN_matrix = np.zeros((d_array_binned.size, d_array_binned.size))

        for n in range(d_array_binned.size):
            HSHpN_matrix[n] = np.roll(np.roll(autocorr_fft, n%autocorr_fft.shape[1], axis=1), n//autocorr_fft.shape[1], axis=0).flatten()

        j_masked = masked_j()
        HSHpN_matrix[j_masked] = 0.0
        HSHpN_matrix[:, j_masked] = 0.0
        HSHpN_matrix[np.diag_indices(d_array_binned.size)] += var_n
        HSHpN_matrix[j_masked, j_masked] = autocorr_fft[0, 0]

        # Solve the dense system with conjugate gradients.
        x, exit_code = cg(HSHpN_matrix, d_array_binned.flatten(), maxiter=HSHpN_matrix.size)

        if exit_code != 0:
            print(f"Warning: the conjugate-gradient solver did not converge (exit code {exit_code}).")

        del HSHpN_matrix
        gc.collect()

        SH_matrix = np.zeros((d_array_binned.size, d_array_binned.size))
        for n in range(d_array_binned.size):
            SH_matrix[n] = np.roll(np.roll(autocorr_fft, n%autocorr_fft.shape[1], axis=1), n//autocorr_fft.shape[1], axis=0).flatten()
        SH_matrix[:,j_masked] = 0.0

        # Apply the solved field through the signal covariance.
        s_WF = SH_matrix.dot(x).reshape(d_array_binned.shape)

        del SH_matrix
        gc.collect()

        if NAME_KEY!='':
            np.savetxt('s_WF_'+NAME_KEY+'.dat', s_WF)
        if verbose:
            print(
                f"Wiener reconstruction complete: shape={d_array_binned.shape}, "
                f"noise ratio={var_n / autocorr_fft[0, 0]:.3e}, "
                f"elapsed={time.time() - start:.1f} s.")

        return s_WF
