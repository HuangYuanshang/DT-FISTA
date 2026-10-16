"""Generate the DT-FISTA flowchart for selected algorithm cycles.

Run from this directory with ``python dtfista_real_flowchart.py``.
NPZ inputs are read from ``data/inputs/`` and the figure is written to ``figures/``.
"""
from pathlib import Path
import os
import sys
import inspect

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DATA / ".matplotlib"))
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
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

from scipy.ndimage import label, find_objects

def make_trace():
    dyn, mask, ts, freqs = load_observation('B1237+25')
    f0, t0, n_candidates = cd.find_clean_patch_random(mask, 300, seed=0, max_masked_frac=.01)
    original = dyn[f0:f0+300, t0:t0+300].copy()
    rfi = mask[f0:f0+300, t0:t0+300] <= .5
    reference = cd.fill_rfi_interp(original, rfi) if rfi.any() else original.copy()
    valid, _ = cd.make_combined_hole(300, 4500, 0)
    data = reference.copy()
    data[~valid] = reference[valid].mean()
    selected, initial, finish = {}, {}, {}
    last = {}
    pending = None
    lines, start = inspect.getsourcelines(fista_inpaint._fista_core)
    initial_line = next(start+i for i, line in enumerate(lines)
                        if line.strip() == 'while should_continue():')
    # Identify the cycle-end and stopping assignments.
    x_assignment_lines = [start+i for i, line in enumerate(lines)
                           if line.strip() == 'x = x_new']
    momentum_line = next(start+i for i, line in enumerate(lines)
                         if line.strip().startswith('t_new = '))
    completed_x_line, stopping_x_line = x_assignment_lines
    target_code = fista_inpaint._fista_core.__code__

    def trace(frame, event, arg):
        nonlocal last, pending
        if frame.f_code is not target_code:
            return None
        loc = frame.f_locals
        if event == 'line' and frame.f_lineno == initial_line and loc['it'] == 0:
            initial.update(D0=loc['x'].copy(), D_mo0=loc['y'].copy(),
                           tilde_S1=loc['S'].copy(), window=loc['window'],
                           global_mean=loc['global_mean'],
                           thresh_val=loc['thresh_val'])
        # Capture the projected state before momentum extrapolation.
        if event == 'line' and frame.f_lineno == momentum_line:
            pending = {
                'tilde_S': loc['S'].copy(),
                'tilde_S_shrink': loc['S_new'].copy(),
                'D_tilde': loc['x_rec'].copy(),
                'D': loc['x_new'].copy(),
                'D_mo_previous': loc['y'].copy(),
                'D_previous': loc['x'].copy(),
            }
            pending.update(thresh_val=loc['thresh_val'], it=loc['it'])
        # Capture the completed cycle.
        if event == 'line' and frame.f_lineno == completed_x_line and pending is not None:
            state = pending.copy()
            state.update(D_mo=loc['y'].copy(), momentum=loc['momentum'])
            state['it'] = loc['it']
            last = state
            if loc['it'] in (1, 64):
                selected[loc['it']] = state.copy()
            pending = None
        if event == 'line' and frame.f_lineno == stopping_x_line:
            finish.update(tilde_S_stop=loc['S'].copy(), D_mo=loc['y'].copy(),
                          D=loc['x'].copy(), it=loc['it'],
                          thresh_val=loc['thresh_val'], mag=loc['mag'].copy())
        if event == 'return' and 'it' not in finish:
            finish.update(tilde_S_stop=loc['S'].copy(), D_mo=loc['y'].copy(),
                          D=loc['x'].copy(), it=loc['it'],
                          thresh_val=loc['thresh_val'], mag=loc['mag'].copy())
        return trace

    sys.settrace(trace)
    try:
        reconstructed = fista_inpaint.fista_inpaint(data, valid.astype(np.int8), edge_reinpaint=False, verbose=False)
    finally:
        sys.settrace(None)
    K = int(finish['it'])
    selected[K] = last.copy()
    assert K > 64
    assert np.isfinite(reconstructed).all()
    np.testing.assert_array_equal(reconstructed[valid], reference[valid])
    for n, state in selected.items():
        np.testing.assert_allclose(fft2(state['D_mo_previous']), state['tilde_S'], rtol=2e-12, atol=2e-10)
        np.testing.assert_allclose(ifft2(state['tilde_S_shrink']).real, state['D_tilde'], rtol=0, atol=2e-12)
        np.testing.assert_array_equal(state['D'][valid], initial['D0'][valid])
        np.testing.assert_array_equal(state['D_mo'][valid], initial['D0'][valid])
        np.testing.assert_allclose(state['D_mo'], state['D'] +
                                   state['momentum'] * (state['D'] - state['D_previous']),
                                   rtol=0, atol=2e-12)
        np.testing.assert_allclose(state['thresh_val'], initial['thresh_val']*.95**(n-1), rtol=2e-14)
        assert state['tilde_S_shrink'][0, 0] == state['tilde_S'][0, 0]
    assert finish['thresh_val'] < np.median(finish['mag'])
    assert last['thresh_val'] >= np.median(np.abs(last['tilde_S']))

    # Use the actual dynamic-spectrum scale for displayed panels.
    working_reference = (reference-initial['global_mean'])*initial['window']
    dyn_limits = np.percentile(finish['D'], [.5, 99.5])
    # Keep the stopping-test transform separate from cycle K.
    fft_limits = np.percentile(np.abs(last['tilde_S']), [.5, 99.5])
    assert 0 < fft_limits[0] < fft_limits[1]
    payload = dict(D_ori=reference, reference_raw=original,
                   D_submu_han=working_reference,
                   reference_fft=fft2(working_reference), original_rfi_mask=rfi,
                   mask=valid, D_mean_inpainted=data, D0=initial['D0'],
                   D_mo0=initial['D_mo0'], D_mo_final=finish['D_mo'],
                   tilde_S_final=last['tilde_S'], tilde_S_stop=finish['tilde_S_stop'],
                   D_final=finish['D'], D_hat=reconstructed,
                   window=initial['window'],
                   times_min=ts[t0:t0+300], frequencies_MHz=freqs[f0:f0+300])
    for n, st in selected.items():
        for key in ('tilde_S', 'tilde_S_shrink', 'D_tilde', 'D', 'D_mo'):
            payload[f'cycle_{n}_{key}'] = st[key]
    info = dict(pulsar='B1237+25', shape=[300, 300], crop_frequency_index=int(f0), crop_time_index=int(t0),
                candidate_count=n_candidates, original_rfi_pixels=int(rfi.sum()), artificial_mask_pixels=int((~valid).sum()),
                retained_mean=float(initial['global_mean']), iterations=K, decay=.95, init_rank=10,
                threshold_initial=float(initial['thresh_val']), dynamic_percentiles=dyn_limits.tolist(),
                fft_magnitude_percentiles=fft_limits.tolist(),
                fft_scale_reference=rf'abs(tilde S^({K})) in displayed cycle {K}',
                fft_percentiles_include_dc=True, fftshift_for_display_only=True,
                stop_threshold=float(finish['thresh_val']), stop_median=float(np.median(finish['mag'])),
                displayed_cycles={str(n):dict(k=n, threshold=float(s['thresh_val']), beta=float(s['momentum']),
                    nonzero_coefficients=int(np.count_nonzero(np.abs(s['tilde_S_shrink'])))) for n,s in selected.items()},
                display_units='Actual mean-subtracted working arrays from the NPZ input. No dewindowing for D_mo panels.',
                final_D_mo_is_extrapolated=True, D_final_is_working_projection=True,
                D_hat_is_returned_inpainting=True,
                rmse_sigma=float(cd.dyn_mask_rmse(reconstructed, reference, ~valid)))
    return payload, info

def draw_mask_transfers(fig,inverse_ax,mask,xs,pw,y,h):
    """Draw arrows from each masked region in the inverse-transform panel.

    The arrows show where the inverse-transform estimate is inserted before
    the data-consistency projection restores the retained samples.
    """
    components,count=label(~mask)
    assert count==3
    records=[]
    for region,slices in enumerate(find_objects(components),1):
        fy,tx=slices
        x0,x1=tx.start/300,tx.stop/300
        y0,y1=fy.start/300,fy.stop/300
        xc,yc=(x0+x1)/2,(y0+y1)/2
        pts=np.argwhere(components==region)
        is_square=len(pts)/((fy.stop-fy.start)*(tx.stop-tx.start))>.9
        if is_square:
            kind='square';pad=.012;arm=.045
            segments=[]
            for xx,sx in ((x0-pad,1),(x1+pad,-1)):
                for yy,sy in ((y0-pad,1),(y1+pad,-1)):
                    segments.append([(xx+sx*arm,yy),(xx,yy),(xx,yy+sy*arm)])
            source=(x1+pad,yc);target=(x0-.003,yc)
            delta=(xs[3]-xs[2])/pw
            span=delta+target[0]-source[0]
            controls=((source[0]+span/3,yc+.045),
                      (source[0]+2*span/3,yc+.045))
        elif max(x1-x0,y1-y0)<.4:
            kind='circle';arm=.055;pad=.014
            segments=[[(xc-arm/2,y0-pad),(xc+arm/2,y0-pad)],
                      [(xc-arm/2,y1+pad),(xc+arm/2,y1+pad)],
                      [(x0-pad,yc-arm/2),(x0-pad,yc+arm/2)],
                      [(x1+pad,yc-arm/2),(x1+pad,yc+arm/2)]]
            source=(x1+pad,yc);target=(xc,y1+.003)
            controls=((1.03,1.04),(1.72,1.04))
        else:
            kind='irregular'
            # Brackets perpendicular to the descending strip's end faces.
            left=pts[pts[:,1]<tx.start+4].mean(0)[::-1]/300
            right=pts[pts[:,1]>=tx.stop-4].mean(0)[::-1]/300
            direction=(right-left)/np.linalg.norm(right-left)
            normal=np.array([-direction[1],direction[0]])
            segments=[]
            for center,inward in ((left,direction),(right,-direction)):
                base=center-.018*inward
                a,b=base-.052*normal,base+.052*normal
                segments.append([a+.038*inward,a,b,b+.038*inward])
            bottom=pts[pts[:,0]==fy.start][:,1].mean()/300+.5/300
            source=(bottom,y0-.012);target=(bottom,y0-.003)
            controls=((1.05,-.035),(1.99,-.035))
        for segment in segments:
            inverse_ax.plot(*np.array(segment).T,transform=inverse_ax.transAxes,
                color='black',lw=1.05,clip_on=False)
        delta=(xs[3]-xs[2])/pw
        end=np.array([delta+target[0],target[1]])
        if controls:
            vertices=np.array([source,*controls,end],dtype=float)
            codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4]
            t=np.linspace(0,1,1501)[:,None]
            sampled=(1-t)**3*vertices[0]+3*(1-t)**2*t*vertices[1]+3*(1-t)*t*t*vertices[2]+t**3*vertices[3]
        else:
            vertices=np.array([source,end]);codes=[MplPath.MOVETO,MplPath.LINETO]
            sampled=np.linspace(vertices[0],vertices[1],1501)
        # Check the arrow's endpoint stays just outside the target mask, and
        # each curved arrow avoids other mask components in both panels.
        target_pixel=np.floor(np.array(target[::-1])*300).astype(int)
        assert components[tuple(target_pixel)]==0,(kind,target_pixel)
        for offset in (0,delta):
            q=sampled-np.array([offset,0])
            inside=(q[:,0]>=0)&(q[:,0]<1)&(q[:,1]>=0)&(q[:,1]<1)
            ij=np.floor(q[inside,::-1]*300).astype(int)
            if offset==delta:
                assert not np.any(components[ij[:,0],ij[:,1]]==region),kind
            if kind!='square':
                assert not np.any((components[ij[:,0],ij[:,1]]!=0)&(components[ij[:,0],ij[:,1]]!=region)),kind
        vertices[:,0]=xs[2]+pw*vertices[:,0];vertices[:,1]=y+h*vertices[:,1]
        path=MplPath(vertices,codes)
        fig.add_artist(FancyArrowPatch(path=path,transform=fig.transFigure,
            arrowstyle='-|>',mutation_scale=10,lw=.65,color='black',zorder=8))
        records.append(dict(mask_region=kind,endpoint_axes=list(target),endpoint_inside_mask=False,
            curved=controls is not None,linewidth=.65,
            upward_bend_axes=float(sampled[:,1].max()-yc) if kind=='square' else None))
    return records

def plot(p, info):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,
        'axes.titlesize':13,'axes.labelsize':12,'xtick.labelsize':11,
        'ytick.labelsize':11,'pdf.fonttype':42,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(18,13.5),facecolor='white')
    fig.suptitle('DT-FISTA: Fourier-sparse inpainting of a B1237+25 dynamic-spectrum patch',
                 y=.985,fontsize=22)
    info['dynamic_percentiles'] = np.percentile(p['D_final'], [.5, 99.5]).tolist()
    info['dynamic_scale_reference'] = f"D_con^({info['iterations']})"
    dn=Normalize(*info['dynamic_percentiles'])
    sn=LogNorm(*info['fft_magnitude_percentiles'])
    cmap=plt.get_cmap('magma').copy(); cmap.set_bad('#080612'); cmap.set_under('#080612')
    ts,fs=p['times_min'],p['frequencies_MHz']
    extent=[ts[0],ts[-1],fs[0],fs[-1]]
    shown_axes=[]
    data_axes=[]

    def panel(rect,arr,title,label,spectral=False,axes=False,outline=False):
        ax=fig.add_axes(rect)
        ax.set_box_aspect(1)
        data_axes.append(ax)
        if spectral:
            z=np.ma.masked_less_equal(np.abs(fftshift(arr)),0)
            im=ax.imshow(z,origin='lower',aspect='auto',cmap=cmap,norm=sn,
                extent=(-150.5,149.5,-150.5,149.5),interpolation='nearest')
            if axes:
                ax.set_xticks([-150,0,149]); ax.set_yticks([-150,0,149])
                ax.set_xlabel('Doppler bin'); ax.set_ylabel('Delay bin')
        else:
            im=ax.imshow(arr,origin='lower',aspect='auto',cmap='viridis',norm=dn,
                extent=extent,interpolation='nearest')
            if axes:
                ax.set_xlabel('Time (min)'); ax.set_ylabel('Frequency (MHz)')
                ax.locator_params(axis='x',nbins=3); ax.locator_params(axis='y',nbins=3)
            if outline:
                ax.contour(p['mask'].astype(float),levels=[.5],colors='black',linewidths=.8,
                    origin='lower',extent=extent)
        if not axes:
            ax.set_axis_off()
        else:
            shown_axes.append(label)
        ax.set_title(f'({label}) {title}',pad=7)
        return im

    left,width=.063,.119
    # Use equal physical dimensions for data panels.
    h=width*fig.get_figwidth()/fig.get_figheight()
    ys=(.73,.43,.13)
    # The constrained initial state is the first panel.
    initim=panel([left,ys[0],width,h],p['D0'],
        'Initial array ' + r'$D_{\mathrm{mo}}^{(0)}=D_{\mathrm{con}}^{(0)}$' +
        '\nmasked samples set to zero','a',axes=True,outline=True)

    xs=[.258,.405,.552,.699,.846]
    pw=.119
    K=info['iterations']
    cycles=(1,64,K)
    transfers_by_cycle = {}
    for row,(n,y) in enumerate(zip(cycles,ys)):
        cycle=info['displayed_cycles'][str(n)]
        suffix=' (final)' if n==K else ''
        fig.text(xs[0],y+h+.05,f'Cycle {n}{suffix}',fontsize=18)
        fig.text(.68,y+h+.05,rf'$\lambda_{{{n}}}={cycle["threshold"]:.3g}$',fontsize=14)
        states=('tilde_S','tilde_S_shrink','D_tilde','D','D_mo')
        titles=[rf'1. FFT: $|\widetilde{{S}}^{{({n})}}|$',
            rf'2. Shrink: $|\widetilde{{S}}_{{\mathrm{{shrink}}}}^{{({n})}}|$',
            rf'3. Inverse: $\widetilde{{D}}^{{({n})}}$',
            rf'4. Project: $D_{{\mathrm{{con}}}}^{{({n})}}$',
            rf'5. Momentum: $D_{{\mathrm{{mo}}}}^{{({n})}}$']
        for j,(state,title) in enumerate(zip(states,titles)):
            if n==K and j==4:
                fig.text(xs[j]+pw/2,y+h/2,
                    'Output ' + r'$\widehat{D}$:' + '\nundo Hanning' +
                    '\nwindow, add ' + r'$\mu$' + ',' +
                    '\nrestore retained' + '\nsamples from ' +
                    r'$D_{\mathrm{ori}}$.',
                    ha='center',va='center',fontsize=14)
                continue
            panel([xs[j],y,pw,h],p[f'cycle_{n}_{state}'],title,
                chr(98+row*5+j),spectral=j<2)
            if j == 2:
                inverse_ax = data_axes[-1]
            if j<4 and j != 2:
                fig.add_artist(FancyArrowPatch((xs[j]+pw+.004,y+h/2),
                    (xs[j+1]-.005,y+h/2),transform=fig.transFigure,
                    arrowstyle='-|>',mutation_scale=12,lw=1.1,color='.25'))
        transfers_by_cycle[str(n)] = draw_mask_transfers(
            fig, inverse_ax, p['mask'], xs, pw, y, h)
        if row<2:
            next_n=cycles[row+1]
            route_y=y-.06
            # Route to the next displayed cycle.
            vertices=[(xs[-1]+pw+.002,y+h/2),(.979,y+h/2),(.979,route_y),
                      (.240,route_y),(.240,ys[row+1]+h/2),(xs[0]-.003,ys[row+1]+h/2)]
            route=MplPath(vertices,[MplPath.MOVETO]+[MplPath.LINETO]*5)
            fig.add_artist(FancyArrowPatch(path=route,transform=fig.transFigure,
                arrowstyle='-|>',mutation_scale=15,lw=1.8,color='.25'))
            fig.text(.609,route_y+.0035,
                rf'$\lambda_{{k+1}}=0.95\lambda_k$; continue through cycles {n+1}-{next_n-1}',
                ha='center',va='bottom',fontsize=12,color='.3')

    # Connect initialization to the first FFT.
    path=MplPath([(left+width+.004,ys[0]+h/2),(xs[0]-.005,ys[0]+h/2)],
                 [MplPath.MOVETO,MplPath.LINETO])
    fig.add_artist(FancyArrowPatch(path=path,transform=fig.transFigure,arrowstyle='-|>',
        mutation_scale=15,lw=1.1,color='.25'))
    # Place both colorbars below the initial panel.
    fft_sm=plt.cm.ScalarMappable(norm=sn,cmap=cmap)
    fft_sm.set_array([])
    for rect,im,title in [([.015,.44,.205,.012],initim,'Dynamic-spectrum intensity'),
                          ([.015,.34,.205,.012],fft_sm,r'Fourier coefficient magnitude (log scale)')]:
        cb=fig.colorbar(im,cax=fig.add_axes(rect),orientation='horizontal',extend='both')
        cb.set_label(title,fontsize=12,labelpad=2)
        cb.ax.tick_params(labelsize=11,pad=1)
    assert shown_axes==['a']
    # Check rendered text and panel geometry.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    ratios=[ax.get_window_extent(renderer).width/ax.get_window_extent(renderer).height for ax in data_axes]
    np.testing.assert_allclose(ratios,1,rtol=0,atol=1e-12)
    texts=list(fig.texts)
    for ax in fig.axes:
        texts.append(ax.title)
        if ax.axison:
            for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
                texts.extend([axis.label,axis.get_offset_text()])
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if min(limits)<=tick.get_loc()<=max(limits):
                        texts.extend([tick.label1,tick.label2])
    texts=[t for t in texts if t.get_visible() and t.get_text().strip()]
    boxes=[t.get_window_extent(renderer) for t in texts]
    overlaps=[]
    for i,a in enumerate(boxes):
        for j in range(i+1,len(boxes)):
            b=boxes[j]
            if min(a.x1,b.x1)-max(a.x0,b.x0)>.5 and min(a.y1,b.y1)-max(a.y0,b.y0)>.5:
                overlaps.append([texts[i].get_text(),texts[j].get_text()])
    assert not overlaps, f'Rendered text overlaps: {overlaps}'
    info['figure_layout']={'initial_panel_axes':['a'],'other_data_axes_visible':False,
        'cycles':list(cycles),'mask_outline_color':'black','data_panel_count':15,
        'cycle_connections':[[cycles[i],cycles[i+1]] for i in range(2)],
        'all_data_panels_square':True,'data_panel_width_height_ratios':ratios,
        'vertical_panel_gap_inches':(ys[0]-ys[1]-h)*fig.get_figheight(),
        'connector_distance_below_panel_inches':.06*fig.get_figheight(),
        'first_cycle_mask_transfer_arrows':3,'inverse_mask_markers':'square: four L corners; irregular: two end brackets; circle: four short marks',
        'mask_transfer_details':transfers_by_cycle[str(cycles[0])],
        'mask_transfers_by_cycle':transfers_by_cycle,
        'total_mask_transfer_arrows':sum(len(records) for records in transfers_by_cycle.values()),
        'between_cycle_arrow_linewidth':1.8,
        'initialization_arrow_linewidth':1.1,'within_cycle_arrow_linewidth':1.1,
        'final_output_text':r'Output: \widehat{D}; undo the Hanning window, add the mean, and restore retained samples from D_{\mathrm{ori}}.',
        'final_momentum_panel_shown':False,
        'rendered_text_overlap_count':len(overlaps)}
    fig.savefig(OUT/'dtfista_real_flowchart.png',dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    payload, info = make_trace()
    plot(payload, info)
    print(f"Saved {FIGURES / 'dtfista_real_flowchart.png'}; final iteration {info['iterations']}", flush=True)

