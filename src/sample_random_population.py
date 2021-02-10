#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Displays histograms of random populations for a scenario
'''
import argparse, os, time, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen.utils as pmu

try:
    import pm_cpp_ext
except ImportError:
    print('proj_manen.simulated_annealing: failed to import native code')

def mmmm(_v): return np.min(_v), np.max(_v), np.mean(_v), np.median(_v)
def _fmt_mmmm2(_v):
    with np.printoptions(precision=2, suppress=True):
        return 'min/max: {} / {}  mean: {} med: {}'.format(*[np.array2string(__v) for __v in mmmm(_v)])

def sample_solutions(filenames, nb_samples=int(1e5), force_recompute=False, use_native=True, show_2d=True):
    _scens = [pmu.Scenario(filename=_f) for _f in filenames]
    _durs = []
    for _scen in _scens:
        cache_filename = f'/tmp/samples__{_scen.name}__{nb_samples}.npz'
        if force_recompute or not os.path.exists(cache_filename):
            print(f'sampling and storing to {cache_filename}')
            solver = pm_cpp_ext.Solver(_scen.drone, _scen.targets) if use_native else pm
            _durs.append([solver.intercept_sequence_copy(_scen.drone, np.random.permutation(_scen.targets).tolist())[1] for _i in range(nb_samples)])
            np.savez(cache_filename, durs=_durs[-1])
        else:
            print(f'loading samples from {cache_filename}')
            _durs.append(np.load(cache_filename)['durs'])

    nb_scen = len(filenames)
    nr, nc = nb_scen, 2 if show_2d else 1
    fig = plt.figure(tight_layout=True, figsize=[6.40*nc, 2.56*nr]);
    fig.canvas.set_window_title("Random Sampling of Solutions")

    gs = fig.add_gridspec(nr, 3)
    axes_2d = [fig.add_subplot(gs[_i,:1]) for _i in range(nb_scen)]
    axes_histo = [fig.add_subplot(gs[_i,1:]) for _i in range(nb_scen)]

    np.set_printoptions(precision=3)
    for ax, _d, _scen in zip(axes_histo, _durs, _scens):
        _dens, _, _ = ax.hist(_d, label=f'random sampling ({nb_samples:.0e} samples)', density=True)
        for sol_name in ['best', 'hc', 'optimal']:
            try:
                _name, _dur, _seq = _scen.solution_by_name(sol_name)
                #ax.annotate(f'{sol_name}', xy=(_dur, 1))#, xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.05),)
                ax.plot([_dur, _dur],[0, np.max(_dens)*0.8], label=f'{sol_name}: {_dur:.3f}s', alpha=0.6, linewidth=2)
            except KeyError: pass
        pmu.decorate(ax, _fmt_mmmm2(_d), xlab='time in s', legend=True)

    for ax, _scen in zip(axes_2d, _scens):
        pmu.plot_scenario(_scen, title=_scen.name, annotate=False, fig=fig, ax=ax)


    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample random population.')
    parser.add_argument("filename", nargs='*')
    parser.add_argument('-s', '--nb_samples', help='number of samples', type=int, default=int(1e5))
    args = parser.parse_args()
    sample_solutions(args.filename, args.nb_samples)
    plt.show()
