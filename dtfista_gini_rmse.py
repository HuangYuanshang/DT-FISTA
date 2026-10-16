"""Relate Fourier-magnitude concentration to DT-FISTA inpainting error.

Run from this directory with ``python dtfista_gini_rmse.py``.
NPZ inputs are read from ``data/inputs/`` and figures are written to ``figures/``.
"""
from pathlib import Path
import os
import csv
import time
import json

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DATA / ".matplotlib"))
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common_data as cd
import fista_inpaint

N = 300
OUT = FIGURES
SECONDARY_SPECTRUM_COLOR = '#1B9E77'  # E_sec curve color in fista_iteration_rmse_noise.py
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

from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

RESULT_DIR=DATA/"dtfista_gini_results"
RESULT_DIR.mkdir(exist_ok=True)

def counts_for(mask):
    bad=(mask<=.5).astype(np.int32)
    ii=np.pad(bad.cumsum(0).cumsum(1),((1,0),(1,0)))
    return ii[N:,N:]-ii[:-N,N:]-ii[N:,:-N]+ii[:-N,:-N]

def ban(a,f,t):
    a[max(0,f-N+1):f+N,max(0,t-N+1):t+N]=False

def greedy(eligible,rng=None,reverse=(False,False)):
    a=eligible[::(-1 if reverse[0] else 1),::(-1 if reverse[1] else 1)].copy()
    result=[]
    while a.any():
        if rng is None:
            q=int(a.argmax())
        else:
            remaining=np.flatnonzero(a)
            q=int(remaining[rng.integers(len(remaining))])
        f,t=np.unravel_index(q,a.shape)
        ban(a,f,t)
        result.append((a.shape[0]-1-f if reverse[0] else f,a.shape[1]-1-t if reverse[1] else t))
    return result

def bands(eligible):
    """DP over every possible band origin, packing each band left to right."""
    choices=[]
    for row in eligible:
        origins=np.flatnonzero(row)
        selected=[];end=-N
        for t in origins:
            if t>=end+N:selected.append(int(t));end=t
        choices.append(selected)
    length=eligible.shape[0]
    best=np.zeros(length+N+1,dtype=int)
    take=np.zeros(length,dtype=bool)
    for f in range(length-1,-1,-1):
        value=len(choices[f])+best[f+N]
        if value>best[f+1]:best[f]=value;take[f]=True
        else:best[f]=best[f+1]
    result=[];f=0
    while f<length:
        if take[f]:result.extend((f,t) for t in choices[f]);f+=N
        else:f+=1
    return result

def maximal_extend(eligible,selected):
    available=eligible.copy()
    for f,t in selected:ban(available,f,t)
    while available.any():
        f,t=np.unravel_index(available.argmax(),available.shape)
        selected.append((int(f),int(t)));ban(available,f,t)
    return selected

def solve(psr):
    configure_data()
    start=time.perf_counter()
    mask=cd.load_mask(psr)
    counts=counts_for(mask);eligible=counts<450
    if not eligible.any():
        report=dict(n_candidates=0,n_selected=0,n_rfi_min=None,n_rfi_max=None,
            global_optimum_proven=True,strict_mask_limit=.005,maximum_masked_pixels=449,
            no_further_qualifying_patch_can_be_added=True)
        np.savez_compressed(RESULT_DIR/f'{psr}_selection.npz',f0s=np.array([],dtype=int),
            t0s=np.array([],dtype=int),n_rfi=np.array([],dtype=int),
            candidate_pool=np.empty((0,2),dtype=int),eligible_origins=eligible)
        (RESULT_DIR/f'{psr}_selection_search.json').write_text(json.dumps(report,indent=2))
        return report
    attempts=[];pool=set()
    def record(name,points):
        points=maximal_extend(eligible,list(points))
        attempts.append((name,points));pool.update(points)
        
        if not name.startswith('random_'): print(psr,name,len(points),flush=True)
    for rf in (False,True):
        for rt in (False,True):
            record(f'scan_{rf}_{rt}',greedy(eligible,reverse=(rf,rt)))
            transformed=eligible[::(-1 if rf else 1),::(-1 if rt else 1)]
            for transpose in (False,True):
                pts=bands(transformed.T if transpose else transformed)
                if transpose:pts=[(t,f) for f,t in pts]
                pts=[(eligible.shape[0]-1-f if rf else f,eligible.shape[1]-1-t if rt else t) for f,t in pts]
                record(f'bands_{rf}_{rt}_{transpose}',pts)
    best_name,best=max(attempts,key=lambda a:len(a[1]))
    candidates=np.array(sorted(pool),dtype=np.int32)
    # Each conflict constrains two overlapping candidate windows. Binary set
    # packing maximizes the number selected from all candidate packings.
    pairs=[]
    for i,(f,t) in enumerate(candidates):
        js=np.flatnonzero((np.abs(candidates[:i,0]-f)<N)&(np.abs(candidates[:i,1]-t)<N))
        pairs.extend((int(j),i) for j in js)
    pair=np.array(pairs,dtype=np.int32)
    rows=np.repeat(np.arange(len(pair)),2)
    matrix=coo_matrix((np.ones(2*len(pair)),(rows,pair.ravel())),shape=(len(pair),len(candidates))).tocsc()
    result=milp(-np.ones(len(candidates)),integrality=np.ones(len(candidates)),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix,-np.inf,1),options={'time_limit':90,'mip_rel_gap':0})
    optimized=[tuple(map(int,p)) for p in candidates[result.x>.5]] if result.x is not None else []
    if len(optimized)>=len(best):best=optimized;best_name='candidate_pool_milp'
    best=maximal_extend(eligible,list(best))
    f0s,t0s=np.array(sorted(best)).T
    for i in range(len(f0s)):
        assert np.all((np.abs(f0s[:i]-f0s[i])>=N)|(np.abs(t0s[:i]-t0s[i])>=N))
    available=eligible.copy()
    for f,t in zip(f0s,t0s):ban(available,f,t)
    assert not available.any()
    report=dict(n_candidates=int(eligible.sum()),n_selected=len(best),n_rfi_min=int(counts[f0s,t0s].min()),
        n_rfi_max=int(counts[f0s,t0s].max()),best_method=best_name,candidate_pool_size=len(candidates),
        random_starts=0,deterministic_starts=12,all_origins_step=1,
        milp_status=int(result.status),milp_message=result.message,
        pool_best_bound=float(-result.mip_dual_bound) if getattr(result,'mip_dual_bound',None) is not None else None,
        no_further_qualifying_patch_can_be_added=True,global_optimum_proven=False,
        strict_mask_limit=.005,maximum_masked_pixels=449,
        trials=[{'method':name,'count':len(points)} for name,points in attempts],
        elapsed_seconds=time.perf_counter()-start)
    np.savez_compressed(RESULT_DIR/f'{psr}_selection.npz',f0s=f0s,t0s=t0s,n_rfi=counts[f0s,t0s],
        candidate_pool=candidates,eligible_origins=eligible)
    (RESULT_DIR/f'{psr}_selection_search.json').write_text(json.dumps(report,indent=2))
    return report

REGENERATE_SELECTION = False


def selection_manifest_path(psr):
    return RESULT_DIR / f'{psr}_selection_manifest.json'


def save_selection_manifest(psr):
    selection_path = RESULT_DIR / f'{psr}_selection.npz'
    with np.load(selection_path) as selection:
        f0s = selection['f0s'].astype(int).tolist()
        t0s = selection['t0s'].astype(int).tolist()
    manifest = {
        'pulsar': psr,
        'patch_size': N,
        'selection_rule': 'Frozen, validated set of 39 nonoverlapping patch origins used for the reported correlation; each 300x300 patch contains fewer than 0.5% input-mask samples.',
        'origins': [{'f0': f0, 't0': t0} for f0, t0 in zip(f0s, t0s)],
    }
    selection_manifest_path(psr).write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def load_selection_manifest(psr):
    configure_data()
    path = selection_manifest_path(psr)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('pulsar') != psr or int(manifest.get('patch_size', -1)) != N:
        raise ValueError(f'Invalid frozen selection manifest: {path}')
    origins = manifest.get('origins', [])
    if not origins:
        raise ValueError(f'Frozen selection manifest has no origins: {path}')
    f0s = np.array([int(origin['f0']) for origin in origins], dtype=np.int32)
    t0s = np.array([int(origin['t0']) for origin in origins], dtype=np.int32)
    mask = cd.load_mask(psr)
    counts = counts_for(mask)
    if np.any(f0s < 0) or np.any(t0s < 0) or np.any(f0s >= counts.shape[0]) or np.any(t0s >= counts.shape[1]):
        raise ValueError(f'Frozen selection contains out-of-bounds origins: {path}')
    if np.any(counts[f0s, t0s] >= 450):
        raise ValueError(f'Frozen selection violates the 0.5% input-mask limit: {path}')
    for i, (f0, t0) in enumerate(zip(f0s, t0s)):
        if np.any((np.abs(f0s[:i] - f0) < N) & (np.abs(t0s[:i] - t0) < N)):
            raise ValueError(f'Frozen selection contains overlapping patches: {path}')
    np.savez_compressed(RESULT_DIR / f'{psr}_selection.npz',
                        f0s=f0s, t0s=t0s, n_rfi=counts[f0s, t0s],
                        eligible_origins=counts < 450)
    return {
        'n_selected': len(f0s),
        'n_rfi_min': int(counts[f0s, t0s].min()),
        'n_rfi_max': int(counts[f0s, t0s].max()),
        'selection_source': path.relative_to(HERE).as_posix(),
        'random_starts': 0,
        'deterministic': True,
    }


def prepare_selection(psr):
    path = selection_manifest_path(psr)
    if path.is_file() and not REGENERATE_SELECTION:
        return load_selection_manifest(psr)
    report = solve(psr)
    count = improve_selection(psr)
    save_selection_manifest(psr)
    report['n_selected'] = count
    report['selection_source'] = path.relative_to(HERE).as_posix()
    report['deterministic'] = True
    return report

def compute(psr):
    dyn, mask, _, _ = load_observation(psr)
    selection=np.load(RESULT_DIR/f'{psr}_selection.npz')
    f0s,t0s=selection['f0s'],selection['t0s']
    valid_hole,info=cd.make_combined_hole(N,int(.05*N*N),0)
    hole=~valid_hole
    window=cd.make_hann_2d(N,N)
    rows=[]
    start=time.perf_counter()
    for index,(f0,t0) in enumerate(zip(f0s,t0s),1):
        tic=time.perf_counter()
        ref=dyn[f0:f0+N,t0:t0+N].copy()
        original_hole=mask[f0:f0+N,t0:t0+N]<=.5
        n_rfi=int(original_hole.sum())
        assert n_rfi<450
        if n_rfi:
            ref=cd.fill_rfi_interp(ref,original_hole)
        assert np.isfinite(ref).all()
        gini=cd.gini_coefficient(np.abs(np.fft.fft2((ref-ref.mean())*window)))
        damaged=ref.copy()
        damaged[hole]=ref[valid_hole].mean()
        iteration_counts=[]
        def callback(it,reconstruction):
            if it==1:
                iteration_counts.append(0)
            iteration_counts[-1]=it
        filled=fista_inpaint.fista_inpaint(damaged,valid_hole.astype(np.int8),verbose=False,iter_callback=callback)
        assert np.isfinite(filled).all()
        np.testing.assert_array_equal(filled[valid_hole],ref[valid_hole])
        raw_rmse=float(np.sqrt(np.mean((filled[hole]-ref[hole])**2)))
        sigma=float(np.std(ref))
        normalized=cd.dyn_mask_rmse(filled,ref,hole)
        assert sigma>0 and abs(normalized-raw_rmse/sigma)<1e-14
        row=dict(pulsar=psr,patch_id=index,f0=int(f0),t0=int(t0),crop_size=N,
                 original_masked_pixels=n_rfi,original_masked_fraction=n_rfi/(N*N),
                 synthetic_masked_pixels=int(hole.sum()),
                 synthetic_masked_fraction=float(hole.mean()),
                 synthetic_original_mask_overlap=int((hole&original_hole).sum()),
                 gini=float(gini),dynamic_rmse=raw_rmse,reference_std=sigma,
                 normalized_dynamic_rmse=float(normalized),
                 mean_fill_normalized_rmse=cd.dyn_mask_rmse(damaged,ref,hole),
                 primary_iterations=iteration_counts[0],
                 total_iterations=sum(iteration_counts),
                 seconds=time.perf_counter()-tic)
        rows.append(row)
        print(json.dumps(row),flush=True)
        # Overwrite only this run's own progress file after each complete sample.
        (RESULT_DIR/f'{psr}_progress.json').write_text(json.dumps(rows,indent=2))
    with (RESULT_DIR/f'{psr}_samples.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(RESULT_DIR/f'{psr}_samples.npz',
        **{key:np.array([r[key] for r in rows]) for key in rows[0]},
        synthetic_valid_mask=valid_hole,
        input_file=np.array(FILENAMES[psr]),
        definition_gini=np.array('Gini of all 90000 absolute FFT coefficients after subtracting the full reference-patch mean and applying the 10% edge Hann window'),
        definition_rmse=np.array('sqrt(mean((filled[synthetic_mask]-reference[synthetic_mask])**2))/std(reference)'),
        selection=np.array('frozen, validated set of 39 nonoverlapping patch origins; each patch contains <0.5% input-mask samples'))
    print(f'COMPLETE {psr}: n={len(rows)}, seconds={time.perf_counter()-start:.1f}',flush=True)

def summarize(rows):
    x=np.array([r['gini'] for r in rows])
    y=np.array([r['normalized_dynamic_rmse'] for r in rows])
    assert len(x)>=3 and np.isfinite(x).all() and np.isfinite(y).all()
    pear=stats.pearsonr(x,y)
    spear=stats.spearmanr(x,y)
    fit=stats.linregress(x,y)
    return dict(n=len(rows),pearson_r=float(pear.statistic),pearson_p=float(pear.pvalue),
        spearman_rho=float(spear.statistic),spearman_p=float(spear.pvalue),
        linear_slope=float(fit.slope),linear_intercept=float(fit.intercept),
        gini_min=float(x.min()),gini_max=float(x.max()),gini_mean=float(x.mean()),
        normalized_rmse_min=float(y.min()),normalized_rmse_max=float(y.max()),
        normalized_rmse_mean=float(y.mean()),normalized_rmse_median=float(np.median(y)),
        normalized_rmse_std=float(y.std(ddof=1)),
        n_original_interpolated_pixels=sum(r['original_masked_pixels'] for r in rows),
        n_synthetic_original_overlap=sum(r['synthetic_original_mask_overlap'] for r in rows))

def improve_selection(psr):
    """Improve a saved non-overlapping patch selection using local swaps."""
    path=RESULT_DIR/f'{psr}_selection.npz'
    with np.load(path) as src: arrays={k:src[k] for k in src.files}
    eligible=arrays['eligible_origins']
    selected=list(zip(arrays['f0s'],arrays['t0s']))
    while True:
        improved=False
        for index in range(len(selected)):
            rest=selected[:index]+selected[index+1:]
            available=eligible.copy()
            for f,t in rest: ban(available,f,t)
            f,t=np.where(available)
            if not len(f): continue
            if f.max()-f.min()>=N: pair=[(f[0],t[0]),(f[-1],t[-1])]
            elif t.max()-t.min()>=N:
                lo,hi=t.argmin(),t.argmax(); pair=[(f[lo],t[lo]),(f[hi],t[hi])]
            else: continue
            selected=rest+pair; improved=True; break
        if not improved: break
    if selected:
        arrays['f0s'],arrays['t0s']=np.array(sorted(selected)).T
        mask=cd.load_mask(psr)
        counts=counts_for(mask)
        arrays['n_rfi']=counts[arrays['f0s'],arrays['t0s']]
        assert np.all(arrays['n_rfi']<450)
        for i,(f,t) in enumerate(zip(arrays['f0s'],arrays['t0s'])):
            assert np.all((abs(arrays['f0s'][:i]-f)>=N)|(abs(arrays['t0s'][:i]-t)>=N))
    np.savez_compressed(path,**arrays)
    return len(selected)

def format_sci(value, decimals=2):
    mantissa, exponent = f'{value:.{decimals}e}'.split('e')
    return rf'{mantissa}\times 10^{{{int(exponent)}}}'


def plot_gini(rows):
    summary=summarize(rows)
    (RESULT_DIR/'statistics.json').write_text(json.dumps({'B2021+51':summary},indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
    fig,ax=plt.subplots(figsize=(3.35,2.65),layout='constrained')
    ax.set_title('Fourier sparsity and inpainting error\nwithin mask regions',fontsize=9,pad=8)
    ax.scatter([r['gini'] for r in rows],[r['normalized_dynamic_rmse'] for r in rows],s=12,color=SECONDARY_SPECTRUM_COLOR,linewidths=0)
    stats_text = (
        rf'Pearson: $r={summary["pearson_r"]:.3f}$, $p={format_sci(summary["pearson_p"])}$' + '\n' +
        rf'Spearman: $\rho={summary["spearman_rho"]:.3f}$, $p={format_sci(summary["spearman_p"])}$'
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=5.2, linespacing=0.9,
            bbox=dict(facecolor='white',
                      edgecolor='0.6', alpha=0.78, pad=1.2))
    ax.set_xlabel(r'Gini coefficient of $|\mathrm{FFT}|$',fontsize=8)
    ax.set_ylabel(r'$\mathrm{nRMSE}_{\mathrm{dyn}}$',fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[['top','right','left','bottom']].set_visible(True)
    path=FIGURES/'dtfista_gini_rmse.png'
    fig.savefig(path,dpi=300,facecolor='white'); plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Saved {path}',flush=True)

if __name__ == '__main__':
    psr='B2021+51'
    print('Loading the frozen, validated set of 39 nonoverlapping 300x300 patches with <0.5% input mask.',flush=True)
    selection_report=prepare_selection(psr)
    count=selection_report['n_selected']
    if count<3: raise RuntimeError(f'Only {count} qualifying patches; at least three are required.')
    (RESULT_DIR/'selection_summary.json').write_text(json.dumps(selection_report,indent=2),encoding='utf-8')
    print(f'Selected {count} patches; starting DT-FISTA.',flush=True)
    compute(psr)
    rows=json.loads((RESULT_DIR/f'{psr}_progress.json').read_text())
    plot_gini(rows)



