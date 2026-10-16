#!/usr/bin/env python3
"""Benchmark mean filling, Wiener reconstruction, and DT-FISTA on gaps.

The script selects reproducible 300 x 300 patches, inserts three synthetic
gaps, and records accuracy, runtime, and memory measurements. NPZ inputs are
read from ``data/inputs/``; figures and products are written to ``figures/`` and
``data/<pulsar>/``.
"""

import ctypes
import multiprocessing as mp
import os
import platform
import threading
import traceback
import sys

_NUM_THREADS = 1
for i, arg in enumerate(sys.argv):
    if arg == '--threads' and i + 1 < len(sys.argv):
        _NUM_THREADS = int(sys.argv[i + 1])
        break
os.environ["OMP_NUM_THREADS"]       = str(_NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]  = str(_NUM_THREADS)
os.environ["MKL_NUM_THREADS"]       = str(_NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]   = str(_NUM_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(_NUM_THREADS)

import gc
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_data as cd
import fista_inpaint
import wiener_filter

CROP_SIZE = 300
HOLE_FRAC = 0.05
RANDOM_SEED = 0
HOLE_AREA = int(HOLE_FRAC * CROP_SIZE * CROP_SIZE)

# Use a fixed seed so patch selection is reproducible.
PATCH_SEED = 0
# Allow a pulsar-specific masked fraction when no fully clean patch exists.
# B1237+25 requires a 1% tolerance because its RFI is spatially dispersed;
# residual masked samples are linearly interpolated before synthetic gaps are added.
PATCH_MASK_TOL = {'B1237+25': 0.01}


MEMORY_SAMPLE_INTERVALS = {'FISTA': 0.4, 'Wiener': 1000.0}


if os.name == 'nt':
    from ctypes import wintypes

    class _ProcessMemoryCountersEx(ctypes.Structure):
        """Windows process-memory counters used to read the current RSS."""

        _fields_ = [
            ('cb', wintypes.DWORD),
            ('PageFaultCount', wintypes.DWORD),
            ('PeakWorkingSetSize', ctypes.c_size_t),
            ('WorkingSetSize', ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
            ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
            ('PagefileUsage', ctypes.c_size_t),
            ('PeakPagefileUsage', ctypes.c_size_t),
            ('PrivateUsage', ctypes.c_size_t),
        ]

    _get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    _get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    _get_current_process.argtypes = []
    _get_current_process.restype = wintypes.HANDLE
    _get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    _get_process_memory_info.restype = wintypes.BOOL
else:
    _ProcessMemoryCountersEx = None
    _get_current_process = None
    _get_process_memory_info = None


def _current_process_rss_bytes():
    """Return the current resident-set size in bytes when supported."""
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/self/statm', 'r', encoding='ascii') as stream:
                resident_pages = int(stream.read().split()[1])
            return resident_pages * int(os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, IndexError, AttributeError):
            return None

    if os.name == 'nt':
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        ok = _get_process_memory_info(
            _get_current_process(), ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else None

    return None


class _PeakRSSMonitor:
    """Sample process RSS in a background thread during one reconstruction."""

    def __init__(self, interval):
        self.interval = float(interval)
        self.baseline_bytes = None
        self.peak_bytes = None
        self.incremental_peak_bytes = None
        self._stop_event = threading.Event()
        self._thread = None

    def _sample_once(self):
        rss = _current_process_rss_bytes()
        if rss is not None and (self.peak_bytes is None or rss > self.peak_bytes):
            self.peak_bytes = rss

    def _sample_loop(self):
        while not self._stop_event.wait(self.interval):
            self._sample_once()

    def __enter__(self):
        self.baseline_bytes = _current_process_rss_bytes()
        self.peak_bytes = self.baseline_bytes
        if self.baseline_bytes is not None:
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._sample_once()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._sample_once()
        if self.baseline_bytes is not None and self.peak_bytes is not None:
            self.incremental_peak_bytes = max(
                0, self.peak_bytes - self.baseline_bytes)


def _cpu_model_name():
    """Return a human-readable CPU model without adding a dependency."""
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/cpuinfo', 'r', encoding='utf-8', errors='replace') as stream:
                for line in stream:
                    key, separator, value = line.partition(':')
                    if separator and key.strip().lower() in ('model name', 'hardware'):
                        model = value.strip()
                        if model:
                            return model
        except OSError:
            pass

    if os.name == 'nt':
        try:
            import winreg
            key_path = r'HARDWARE\DESCRIPTION\System\CentralProcessor\0'
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                model, _ = winreg.QueryValueEx(key, 'ProcessorNameString')
            if str(model).strip():
                return str(model).strip()
        except (ImportError, OSError):
            pass

    model = platform.processor().strip()
    if not model:
        model = os.environ.get('PROCESSOR_IDENTIFIER', '').strip()
    return model or 'unavailable'


def _format_memory(byte_count):
    """Format a byte count for a human-readable benchmark summary."""
    if byte_count is None:
        return 'unavailable'
    return (f"{int(byte_count)} bytes "
            f"({byte_count / 2**20:.1f} MiB, {byte_count / 2**30:.3f} GiB)")




def _read_memory_from_summary(summary_path, method_name, metric_name):
    """Read a byte-valued memory metric from an existing text summary."""
    prefix = f"{method_name} {metric_name}:"
    try:
        with open(summary_path, 'r', encoding='utf-8') as stream:
            for line in stream:
                if line.startswith(prefix):
                    value = int(line[len(prefix):].strip().split()[0])
                    return value if value >= 0 else None
    except (OSError, ValueError, IndexError):
        pass
    return None

def _read_text_from_summary(summary_path, field_name):
    """Read one text-valued field from an existing summary."""
    prefix = f"{field_name}:"
    try:
        with open(summary_path, 'r', encoding='utf-8') as stream:
            for line in stream:
                if line.startswith(prefix):
                    value = line[len(prefix):].strip()
                    if not value or value.lower() == 'unavailable':
                        return None
                    return value
    except OSError:
        pass
    return None

def _read_elapsed_from_summary(summary_path, method_name):
    """Read a previously recorded elapsed time from the text summary."""
    prefix = f"{method_name} elapsed time:"
    try:
        with open(summary_path, 'r', encoding='utf-8') as stream:
            for line in stream:
                if line.startswith(prefix):
                    elapsed = float(line[len(prefix):].strip().split()[0])
                    return elapsed if np.isfinite(elapsed) else np.nan
    except (OSError, ValueError, IndexError):
        pass
    return np.nan


def wiener_fill(dyn_hole, hole_mask_int):
    """Reconstruct masked samples with an edge-tapered Wiener estimate."""
    N = dyn_hole.shape[0]
    mean_dyn_hole = np.mean(dyn_hole)
    dyn_hole_centered = dyn_hole - mean_dyn_hole

    # Taper 10% of each edge to suppress FFT boundary ringing.
    edge_frac = 0.1
    edge_f = max(1, int(edge_frac * N))
    edge_t = max(1, int(edge_frac * N))
    _hf = np.hanning(2 * edge_f)
    _wf = np.ones(N)
    _wf[:edge_f] = _hf[:edge_f]; _wf[-edge_f:] = _hf[edge_f:]
    _ht = np.hanning(2 * edge_t)
    _wt = np.ones(N)
    _wt[:edge_t] = _ht[:edge_t]; _wt[-edge_t:] = _ht[edge_t:]
    window_2d = np.outer(_wf, _wt)
    window_2d[window_2d <= 0] = np.min(window_2d[window_2d > 0])

    dyn_hole_centered = dyn_hole_centered * window_2d
    dyn_wiener_centered = wiener_filter.wiener_filter(
        dyn_hole_centered, hole_mask_int, verbose=False)

    invalid_idx = (hole_mask_int < 0.5)

    # Undo the edge taper before restoring the mean level.
    dyn_wiener_centered = dyn_wiener_centered / window_2d
    dyn_wiener_0 = dyn_wiener_centered + mean_dyn_hole

    # Clip unstable edge estimates using statistics from valid central samples.
    clip_edge = max(1, int(edge_frac * N))
    edge_region = np.zeros((N, N), dtype=bool)
    edge_region[:clip_edge, :] = True
    edge_region[-clip_edge:, :] = True
    edge_region[:, :clip_edge] = True
    edge_region[:, -clip_edge:] = True

    center_valid = (~edge_region) & (hole_mask_int > 0.5)
    clip_mean = np.mean(dyn_wiener_0[center_valid])
    clip_std  = np.std(dyn_wiener_0[center_valid])
    clip_upper = clip_mean + 3.0 * clip_std
    dyn_wiener_0[edge_region] = np.clip(dyn_wiener_0[edge_region], 0.0, clip_upper)

    dyn_wiener = dyn_hole.copy()
    dyn_wiener[invalid_idx] = dyn_wiener_0[invalid_idx]
    return dyn_wiener


def _reconstruction_worker(send_connection, method_name, dyn_hole,
                           hole_mask_int, fista_edge_reinpaint):
    """Run one reconstruction and its memory monitor in a fresh process."""
    try:
        gc.collect()
        with _PeakRSSMonitor(MEMORY_SAMPLE_INTERVALS[method_name]) as memory_monitor:
            start_time = time.perf_counter()
            if method_name == 'FISTA':
                reconstruction = fista_inpaint.fista_inpaint(
                    dyn_hole, hole_mask_int,
                    edge_reinpaint=fista_edge_reinpaint,
                    verbose=False)
            elif method_name == 'Wiener':
                reconstruction = wiener_fill(dyn_hole, hole_mask_int)
            else:
                raise ValueError(f"Unknown reconstruction method: {method_name}")
            elapsed = time.perf_counter() - start_time

        send_connection.send({
            'status': 'ok',
            'reconstruction': reconstruction,
            'elapsed': elapsed,
            'memory_sample_interval': memory_monitor.interval,
            'worker_pid': os.getpid(),
            'baseline_rss': memory_monitor.baseline_bytes,
            'peak_rss': memory_monitor.peak_bytes,
            'incremental_peak_rss': memory_monitor.incremental_peak_bytes,
            'cpu_model': _cpu_model_name(),
            'logical_cpu_count': os.cpu_count(),
            'thread_count': _NUM_THREADS,
        })
    except BaseException:
        send_connection.send({
            'status': 'error',
            'method': method_name,
            'traceback': traceback.format_exc(),
        })
    finally:
        send_connection.close()


def _run_reconstruction_process(method_name, dyn_hole, hole_mask_int,
                                fista_edge_reinpaint=False):
    """Run one method in an isolated spawned process and return its results."""
    context = mp.get_context('spawn')
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_reconstruction_worker,
        args=(send_connection, method_name, dyn_hole, hole_mask_int,
              fista_edge_reinpaint),
        name=f'{method_name.lower()}-reconstruction')
    process.start()
    send_connection.close()

    try:
        payload = receive_connection.recv()
    except EOFError:
        payload = {
            'status': 'error',
            'method': method_name,
            'traceback': 'The worker exited without returning benchmark data.',
        }
    finally:
        receive_connection.close()
        process.join()

    if payload.get('status') != 'ok':
        detail = payload.get('traceback', 'No worker error details were returned.')
        raise RuntimeError(
            f"{method_name} reconstruction failed in its worker process:\n{detail}")
    if process.exitcode != 0:
        raise RuntimeError(
            f"{method_name} worker exited with code {process.exitcode}.")
    return payload


def main(pulsar='B2021+51', fista_edge_reinpaint=False):
    """Run the comparison for one pulsar.

    Set ``fista_edge_reinpaint`` to ``False`` to use only the primary FISTA
    pass, or ``True`` to enable the optional edge-refinement pass.
    """
    print(
        f"[{pulsar}] Starting the FISTA and Wiener comparison for "
        f"{CROP_SIZE} x {CROP_SIZE} patches.")

    # 1. Load the full dynamic spectrum and RFI mask.
    dyn, mask, ts, freqs = cd.load_all(pulsar)

    # 2. Select a reproducible patch subject to the pulsar-specific RFI tolerance.
    patch_tol = PATCH_MASK_TOL.get(pulsar, 0.0)
    f0, t0, n_clean = cd.find_clean_patch_random(mask, CROP_SIZE, seed=PATCH_SEED,
                                                 max_masked_frac=patch_tol)
    f1, t1 = f0 + CROP_SIZE, t0 + CROP_SIZE

    dyn_crop   = dyn[f0:f1, t0:t1].copy()
    ts_crop    = ts[t0:t1]
    freqs_crop = freqs[f0:f1]

    # Interpolate residual RFI samples before treating the patch as ground truth.
    n_rfi_patch = 0
    if patch_tol > 0:
        rfi_mask = (mask[f0:f1, t0:t1] <= 0.5)
        n_rfi_patch = int(np.sum(rfi_mask))
        if n_rfi_patch > 0:
            dyn_crop = cd.fill_rfi_interp(dyn_crop, rfi_mask)
            print(
                f"[{pulsar}] Interpolated {n_rfi_patch} RFI-masked samples "
                f"({n_rfi_patch/CROP_SIZE**2*100:.2f}% of the patch).")

    # 3. Add the three synthetic gaps.
    hole_mask_combined, hole_info = cd.make_combined_hole(CROP_SIZE, HOLE_AREA, RANDOM_SEED)
    hole_mask_int = hole_mask_combined.astype(np.int8)
    invalid_idx = (hole_mask_int < 0.5)
    n_hole = int(np.sum(invalid_idx))

    dyn_hole = dyn_crop.copy()
    dyn_hole[~hole_mask_combined] = np.mean(dyn_crop[hole_mask_int > 0.5])

    # 4. Reconstruct the missing samples with FISTA.
    print(
        f"[{pulsar}] Reconstructing {n_hole} synthetic missing samples "
        "with FISTA and Wiener filtering.")
    fista_run = _run_reconstruction_process(
        'FISTA', dyn_hole, hole_mask_int,
        fista_edge_reinpaint=fista_edge_reinpaint)
    dyn_fista = fista_run['reconstruction']
    dt_fista = fista_run['elapsed']
    fista_baseline_rss = fista_run['baseline_rss']
    fista_peak_rss = fista_run['peak_rss']
    fista_incremental_peak_rss = fista_run['incremental_peak_rss']
    fista_cpu_model = fista_run['cpu_model']
    fista_logical_cpu_count = fista_run['logical_cpu_count']
    fista_thread_count = fista_run['thread_count']
    fista_sample_interval = fista_run['memory_sample_interval']

    # 5. Reuse cached Wiener output to avoid repeating the hour-scale solve.
    npz_path = os.path.join(cd.pulsar_outdir(pulsar), 'patch_data.npz')
    info_txt = os.path.join(
        cd.pulsar_outdir(pulsar), 'patch_reconstruction_summary.txt')
    dyn_wiener = None
    dt_wien = np.nan
    wiener_baseline_rss = None
    wiener_peak_rss = None
    wiener_incremental_peak_rss = None
    wiener_cpu_model = None
    wiener_logical_cpu_count = None
    wiener_thread_count = None
    wiener_measurement_mode = None
    cached_measurement_mode = None
    wiener_sample_interval = None
    wiener_cached = False
    if os.path.exists(npz_path) and '--recompute-wiener' not in sys.argv:
        try:
            cached_measurement_mode = _read_text_from_summary(
                info_txt, 'Wiener resource measurement mode')
            interval_text = _read_text_from_summary(info_txt, 'Wiener memory sample interval')
            if interval_text is not None:
                wiener_sample_interval = float(interval_text.split()[0])
            with np.load(npz_path) as _d:
                if ('dyn_wiener' in _d.files
                        and 'hole_mask' in _d.files
                        and _d['dyn_wiener'].shape == dyn_hole.shape
                        and int(_d['crop_region'][0]) == f0
                        and int(_d['crop_region'][2]) == t0
                        and np.array_equal(_d['hole_mask'], hole_mask_int)
                        and cached_measurement_mode == 'isolated spawned process'
                        and wiener_sample_interval == MEMORY_SAMPLE_INTERVALS['Wiener']
                        and all(_read_memory_from_summary(info_txt, 'Wiener', metric) is not None
                                for metric in ('baseline process RSS', 'peak process RSS',
                                               'incremental peak RSS'))):
                    dyn_wiener = _d['dyn_wiener']
                    wiener_cached = True
                    dt_wien = _read_elapsed_from_summary(info_txt, 'Wiener')
        except Exception:
            dyn_wiener = None
            wiener_cached = False
    if dyn_wiener is not None:
        print(f"[{pulsar}] Using the cached Wiener reconstruction.")
        wiener_measurement_mode = cached_measurement_mode
        if wiener_measurement_mode == 'isolated spawned process':
            wiener_baseline_rss = _read_memory_from_summary(
                info_txt, 'Wiener', 'baseline process RSS')
            wiener_peak_rss = _read_memory_from_summary(
                info_txt, 'Wiener', 'peak process RSS')
            wiener_incremental_peak_rss = _read_memory_from_summary(
                info_txt, 'Wiener', 'incremental peak RSS')
            wiener_cpu_model = _read_text_from_summary(
                info_txt, 'Wiener CPU model')
            wiener_logical_cpu_count = _read_text_from_summary(
                info_txt, 'Wiener logical CPUs visible')
            wiener_thread_count = _read_text_from_summary(
                info_txt, 'Wiener configured numerical threads')
    else:
        wiener_run = _run_reconstruction_process(
            'Wiener', dyn_hole, hole_mask_int,
            fista_edge_reinpaint=False)
        dyn_wiener = wiener_run['reconstruction']
        dt_wien = wiener_run['elapsed']
        wiener_sample_interval = wiener_run['memory_sample_interval']
        wiener_baseline_rss = wiener_run['baseline_rss']
        wiener_peak_rss = wiener_run['peak_rss']
        wiener_incremental_peak_rss = wiener_run['incremental_peak_rss']
        wiener_cpu_model = wiener_run['cpu_model']
        wiener_logical_cpu_count = wiener_run['logical_cpu_count']
        wiener_thread_count = str(wiener_run['thread_count'])
        wiener_measurement_mode = 'isolated spawned process'

    # 6. Compute reconstruction metrics.
    hann_win = cd.make_hann_2d(CROP_SIZE, CROP_SIZE)
    sec_orig_p  = cd.secondary_spectrum_power(dyn_crop,   hann_win)
    sec_hole_p  = cd.secondary_spectrum_power(dyn_hole,   hann_win)
    sec_fista_p = cd.secondary_spectrum_power(dyn_fista,  hann_win)
    sec_wien_p  = cd.secondary_spectrum_power(dyn_wiener, hann_win)

    # Dynamic-spectrum RMSE is evaluated only inside the synthetic gaps.
    rmse_dyn_fista = cd.dyn_mask_rmse(dyn_fista,  dyn_crop, invalid_idx)
    rmse_dyn_wien  = cd.dyn_mask_rmse(dyn_wiener, dyn_crop, invalid_idx)
    rmse_dyn_hole  = cd.dyn_mask_rmse(dyn_hole,   dyn_crop, invalid_idx)
    # Secondary-spectrum RMSE is computed in linear power, normalized by the
    # original peak, and reported as log10(RMSE / peak).
    rmse_sec_fista = cd.sec_log10_rmse(sec_fista_p, sec_orig_p)
    rmse_sec_wien  = cd.sec_log10_rmse(sec_wien_p,  sec_orig_p)
    rmse_sec_hole  = cd.sec_log10_rmse(sec_hole_p,  sec_orig_p)
    print(
        f"[{pulsar}] Mask-region dynamic-spectrum RMSE / sigma: "
        f"Mean={rmse_dyn_hole:.4f}, Wiener={rmse_dyn_wien:.4f}, "
        f"FISTA={rmse_dyn_fista:.4f}.")
    print(
        f"[{pulsar}] log10 secondary-spectrum RMSE / maximum: "
        f"Mean={rmse_sec_hole:.4f}, Wiener={rmse_sec_wien:.4f}, "
        f"FISTA={rmse_sec_fista:.4f}.")

    # Estimate noise from four corner regions, each spanning 1/4 x 1/4.
    noise_orig  = cd.sec_noise_db(sec_orig_p)
    noise_hole  = cd.sec_noise_db(sec_hole_p)
    noise_wien  = cd.sec_noise_db(sec_wien_p)
    noise_fista = cd.sec_noise_db(sec_fista_p)
    print(
        f"[{pulsar}] P_noise (dB): "
        f"Original={noise_orig:.2f}, Mean={noise_hole:.2f}, "
        f"Wiener={noise_wien:.2f}, FISTA={noise_fista:.2f}.")

    # Save numerical products consumed by the downstream analyses.
    npz_path = os.path.join(cd.pulsar_outdir(pulsar), 'patch_data.npz')
    np.savez(npz_path,
             dyn_crop=dyn_crop, dyn_hole=dyn_hole,
             dyn_fista=dyn_fista, dyn_wiener=dyn_wiener,
             hole_mask=hole_mask_int,
             crop_region=np.array([f0, f1, t0, t1]),
             ts_crop=ts_crop, freqs_crop=freqs_crop,
             crop_size=CROP_SIZE, hole_frac=HOLE_FRAC, seed=RANDOM_SEED,
             pulsar=np.array(pulsar), patch_mask_tol=patch_tol,
             fista_elapsed_seconds=dt_fista, wiener_elapsed_seconds=dt_wien,
             fista_memory_sample_interval_seconds=fista_sample_interval,
             wiener_memory_sample_interval_seconds=wiener_sample_interval,
             fista_baseline_rss_bytes=fista_baseline_rss if fista_baseline_rss is not None else -1,
             fista_peak_rss_bytes=fista_peak_rss if fista_peak_rss is not None else -1,
             fista_incremental_peak_rss_bytes=fista_incremental_peak_rss if fista_incremental_peak_rss is not None else -1,
             wiener_baseline_rss_bytes=wiener_baseline_rss if wiener_baseline_rss is not None else -1,
             wiener_peak_rss_bytes=wiener_peak_rss if wiener_peak_rss is not None else -1,
             wiener_incremental_peak_rss_bytes=wiener_incremental_peak_rss if wiener_incremental_peak_rss is not None else -1,
             resource_measurement_mode='isolated spawned process',
             memory_unit='bytes; -1 means unavailable', 
             n_rfi_interp=n_rfi_patch,
             fista_edge_reinpaint=np.array(fista_edge_reinpaint),
             rmse_dyn_fista=rmse_dyn_fista, rmse_dyn_wien=rmse_dyn_wien,
             rmse_dyn_hole=rmse_dyn_hole,
             rmse_sec_fista=rmse_sec_fista, rmse_sec_wien=rmse_sec_wien,
             rmse_sec_hole=rmse_sec_hole,
             noise_orig_db=noise_orig, noise_hole_db=noise_hole,
             noise_wien_db=noise_wien, noise_fista_db=noise_fista,
             sec_rmse_unit='log10(sqrt(mean((sec_filled-sec_orig)^2))/DC(sec_orig)), linear power, no dB',
             dyn_rmse_unit='sqrt(mean((filled[hole]-orig[hole])^2))/std(orig)',
             noise_unit='10*log10(mean(4 corners each N/4*N/4 of sec power)/DC(sec power)) [dB]')

    # Write a human-readable summary for figure provenance.
    tol_desc_full = ("clean (0 masked pixels)" if patch_tol == 0
                     else f"masked ≤ {patch_tol*100:.1f}% "
                          f"({n_rfi_patch} px interp-filled)")
    if np.isfinite(dt_wien):
        wiener_elapsed_text = f"{dt_wien:.3f} s"
        wiener_timing_source = (
            "preserved from the existing summary"
            if wiener_cached else "measured in the current run")
    else:
        wiener_elapsed_text = "unavailable"
        wiener_timing_source = "legacy cache without timing metadata"
    if wiener_cached:
        if (wiener_measurement_mode == 'isolated spawned process'
                and wiener_baseline_rss is not None
                and wiener_peak_rss is not None
                and wiener_incremental_peak_rss is not None
                and wiener_cpu_model is not None):
            wiener_resource_source = "preserved from the existing summary"
        else:
            wiener_resource_source = (
                "cached reconstruction predates resource monitoring; "
                "rerun with --recompute-wiener")
    else:
        wiener_resource_source = "measured in the current run"
    wiener_cpu_model_text = wiener_cpu_model or "unavailable"
    wiener_logical_cpu_count_text = wiener_logical_cpu_count or "unavailable"
    wiener_thread_count_text = wiener_thread_count or "unavailable"
    wiener_measurement_mode_text = wiener_measurement_mode or "unavailable"
    with open(info_txt, 'w', encoding='utf-8') as ft:
        ft.write(f"Patch Reconstruction Summary — {pulsar}\n"
                 f"{'='*60}\n"
                 f"Patch: {CROP_SIZE}×{CROP_SIZE} at [f {f0}:{f1}, t {t0}:{t1}] "
                 f"({freqs_crop[0]:.2f}–{freqs_crop[-1]:.2f} MHz, "
                 f"{ts_crop[0]:.2f}–{ts_crop[-1]:.2f} min)\n"
                 f"Patch condition: {tol_desc_full}, "
                 f"{n_clean} candidates, random seed={PATCH_SEED}\n"
                 f"Artificial holes: circle + irregular strip + square, "
                 f"each {HOLE_FRAC*100:.2f}% ({HOLE_AREA} px), "
                 f"hole seed={RANDOM_SEED}, total {n_hole} px "
                 f"({n_hole/CROP_SIZE**2*100:.2f}%)\n"
                 f"FISTA: default params (k=0.95), edge refinement "
                 f"{'enabled' if fista_edge_reinpaint else 'disabled'}\n"
                 f"FISTA elapsed time: {dt_fista:.3f} s (measured in the current run)\n"
                 f"Wiener elapsed time: {wiener_elapsed_text} ({wiener_timing_source})\n"
                 f"\nBenchmark resources:\n"
                 f"  Memory definition: process resident-set size (RSS), sampled every "
                 f"{MEMORY_SAMPLE_INTERVALS['FISTA']:.3f} s for FISTA and "
                 f"{MEMORY_SAMPLE_INTERVALS['Wiener']:.3f} s for Wiener; incremental peak is relative "
                 f"to RSS immediately before each method\n"
                 f"FISTA resource measurement mode: isolated spawned process\n"
                 f"FISTA memory sample interval: {fista_sample_interval:.3f} s\n"
                 f"FISTA CPU model: {fista_cpu_model}\n"
                 f"FISTA logical CPUs visible: "
                 f"{fista_logical_cpu_count if fista_logical_cpu_count is not None else 'unavailable'}\n"
                 f"FISTA configured numerical threads: {fista_thread_count}\n"
                 f"FISTA baseline process RSS: {_format_memory(fista_baseline_rss)}\n"
                 f"FISTA peak process RSS: {_format_memory(fista_peak_rss)}\n"
                 f"FISTA incremental peak RSS: "
                 f"{_format_memory(fista_incremental_peak_rss)}\n"
                 f"FISTA resource source: measured in the current run\n"
                 f"Wiener resource measurement mode: {wiener_measurement_mode_text}\n"
                 f"Wiener memory sample interval: {wiener_sample_interval:.3f} s\n"
                 f"Wiener CPU model: {wiener_cpu_model_text}\n"
                 f"Wiener logical CPUs visible: {wiener_logical_cpu_count_text}\n"
                 f"Wiener configured numerical threads: {wiener_thread_count_text}\n"
                 f"Wiener baseline process RSS: {_format_memory(wiener_baseline_rss)}\n"
                 f"Wiener peak process RSS: {_format_memory(wiener_peak_rss)}\n"
                 f"Wiener incremental peak RSS: "
                 f"{_format_memory(wiener_incremental_peak_rss)}\n"
                 f"Wiener resource source: {wiener_resource_source}\n"
                  f"\nnRMSE_dyn (mask region):\n"
                 f"  Mean={rmse_dyn_hole:.4f}, Wiener={rmse_dyn_wien:.4f}, "
                 f"FISTA={rmse_dyn_fista:.4f}\n"
                  f"E_sec:\n"
                 f"  Mean={rmse_sec_hole:.4f}, Wiener={rmse_sec_wien:.4f}, "
                 f"FISTA={rmse_sec_fista:.4f}\n"
                  f"P_noise (dB):\n"
                 f"  Orig={noise_orig:.2f}, Mean={noise_hole:.2f}, "
                 f"Wiener={noise_wien:.2f}, FISTA={noise_fista:.2f}\n")

    # Retain the dB arrays used by the combined two-pulsar figure.
    sec_orig  = cd.power_to_db_center(sec_orig_p)
    sec_hole  = cd.power_to_db_center(sec_hole_p)
    sec_fista = cd.power_to_db_center(sec_fista_p)
    sec_wien  = cd.power_to_db_center(sec_wien_p)

    wiener_console_time = (
        f"{dt_wien:.1f} s" if np.isfinite(dt_wien) else "unavailable")
    print(
        f"[{pulsar}] Synthetic-gap reconstruction analysis complete "
        f"(FISTA {dt_fista:.1f} s; Wiener {wiener_console_time}). "
        f"Data: {npz_path}; summary: {info_txt}.")
    return {
        'dyn_crop': dyn_crop, 'dyn_hole': dyn_hole,
        'dyn_wiener': dyn_wiener, 'dyn_fista': dyn_fista,
        'sec_orig': sec_orig, 'sec_hole': sec_hole,
        'sec_wien': sec_wien, 'sec_fista': sec_fista,
        'ts_crop': ts_crop, 'freqs_crop': freqs_crop,
        'hole_info': hole_info,
        'rmse_dyn_hole': rmse_dyn_hole, 'rmse_dyn_wien': rmse_dyn_wien,
        'rmse_dyn_fista': rmse_dyn_fista,
        'rmse_sec_hole': rmse_sec_hole, 'rmse_sec_wien': rmse_sec_wien,
        'rmse_sec_fista': rmse_sec_fista,
        'noise_orig': noise_orig, 'noise_hole': noise_hole,
        'noise_wien': noise_wien, 'noise_fista': noise_fista,
        'fista_edge_reinpaint': fista_edge_reinpaint,
    }


def plot_combined(results):
    """Create the 4 x 4 reconstruction comparison for both pulsars."""
    ordered = cd.ordered_pulsar_results(results, 'Synthetic-gap reconstruction analysis')
    fig, axes = plt.subplots(
        4, 4, figsize=(18, 21), sharey='row', constrained_layout=True,
        gridspec_kw={'wspace': 0.0, 'hspace': 0.08})
    fig.set_constrained_layout_pads(w_pad=0.01, wspace=0.0)

    for pulsar_index, (pulsar, result) in enumerate(ordered):
        dyn_row = 2 * pulsar_index
        sec_row = dyn_row + 1
        dyn_crop = result['dyn_crop']
        ts_crop = result['ts_crop']
        freqs_crop = result['freqs_crop']

        # Scale each pulsar independently while applying one scale across methods.
        vmin, vmax = np.nanpercentile(dyn_crop, [0.5, 99.5])
        vmin = 0.0
        sec_vmin, sec_vmax = np.nanpercentile(result['sec_orig'].ravel(), [50, 99.9])
        sec_vmin_disp = sec_vmin - 5

        ts_crop_s = ts_crop * 60.0
        ft_edges, t_edges = cd.make_sec_edges(
            CROP_SIZE, CROP_SIZE, freqs_crop, ts_crop_s)

        dyn_titles = [
            "Original\nDynamic Spectrum",
            f"Mean-Filled\nDynamic Spectrum\n$\\mathrm{{nRMSE}}_{{\\mathrm{{dyn}}}}$ = {result['rmse_dyn_hole']:.3f}",
            f"Wiener\nDynamic Spectrum\n$\\mathrm{{nRMSE}}_{{\\mathrm{{dyn}}}}$ = {result['rmse_dyn_wien']:.3f}",
            f"FISTA\nDynamic Spectrum\n$\\mathrm{{nRMSE}}_{{\\mathrm{{dyn}}}}$ = {result['rmse_dyn_fista']:.3f}",
        ]
        dyn_arrs = [result['dyn_crop'], result['dyn_hole'],
                    result['dyn_wiener'], result['dyn_fista']]
        sec_arrs = [result['sec_orig'], result['sec_hole'],
                    result['sec_wien'], result['sec_fista']]
        sec_titles = [
            f"Original\nSecondary Spectrum\n$P_{{\\mathrm{{noise}}}}$ = {result['noise_orig']:.2f} dB",
            f"Mean-Filled\nSecondary Spectrum\n"
            f"$E_{{\\mathrm{{sec}}}}$ = {result['rmse_sec_hole']:.2f}\n"
            f"$P_{{\\mathrm{{noise}}}}$ = {result['noise_hole']:.2f} dB",
            f"Wiener\nSecondary Spectrum\n"
            f"$E_{{\\mathrm{{sec}}}}$ = {result['rmse_sec_wien']:.2f}\n"
            f"$P_{{\\mathrm{{noise}}}}$ = {result['noise_wien']:.2f} dB",
            f"FISTA\nSecondary Spectrum\n"
            f"$E_{{\\mathrm{{sec}}}}$ = {result['rmse_sec_fista']:.2f}\n"
            f"$P_{{\\mathrm{{noise}}}}$ = {result['noise_fista']:.2f} dB",
        ]

        for col, (title, dyn_p) in enumerate(zip(dyn_titles, dyn_arrs)):
            ax = axes[dyn_row, col]
            im_dyn = ax.pcolormesh(
                ts_crop, freqs_crop, dyn_p, shading='auto', cmap='viridis',
                norm=Normalize(vmin=vmin, vmax=vmax))
            if col == 0:
                cd.draw_hole_markers(
                    ax, result['hole_info'], ts_crop, freqs_crop, CROP_SIZE)
            ax.set_title(f"[{pulsar}] {title}", fontsize=12)
            ax.set_box_aspect(1)
            ax.set_xlabel("Time (min)", fontsize=13)
            if col == 0:
                ax.set_ylabel("Freq (MHz)", fontsize=13)
        cbar_dyn = fig.colorbar(
            im_dyn, ax=axes[dyn_row, :].tolist(), fraction=0.018, pad=0.015,
            label="Intensity")
        cd.style_colorbar(cbar_dyn, tick_labelsize=11)

        for col, (title, sec_p) in enumerate(zip(sec_titles, sec_arrs)):
            ax = axes[sec_row, col]
            im_sec = ax.pcolormesh(
                t_edges, ft_edges, sec_p, shading='auto', cmap='inferno',
                norm=Normalize(vmin=sec_vmin_disp, vmax=sec_vmax))
            ax.set_title(f"[{pulsar}] {title}", fontsize=12)
            ax.set_box_aspect(1)
            ax.set_xlabel("Doppler (mHz)", fontsize=13)
            if col == 0:
                ax.set_ylabel(r"Delay ($\mu$s)", fontsize=13)
        cbar_sec = fig.colorbar(
            im_sec, ax=axes[sec_row, :].tolist(), fraction=0.018, pad=0.015,
            label="dB")
        cd.style_colorbar(cbar_sec, tick_labelsize=11)

    cd.style_subplot_grid(axes, tick_labelsize=11, panel_labelsize=14)
    fig.suptitle(
        "Comparison of Mean, Wiener, and DT-FISTA Inpainting for Artificial Masks",
        fontsize=18)
    out_png = cd.combined_figure_path('patch_compare.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Combined synthetic-gap reconstruction figure saved to {out_png}.")


if __name__ == "__main__":
    mp.freeze_support()
    _results = {}
    for _psr in cd.PULSAR_LIST:
        try:
            _results[_psr] = main(_psr)
        except Exception as e:
            print(f"Warning: synthetic-gap reconstruction analysis failed for {_psr}: {e}")
    plot_combined(_results)
