#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Unit test for simulated annealing
'''
import os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.animations as pma, proj_manen.simulated_annealing as pm_sa
import proj_manen.native_core as pm_nc


def search_heuristic_closest_refined(scen, ntest, debug=False, Tf=None, start_seq_name='heuristic'):
    try: # compute closest_target heuristic if unknown
        name, heur_dur, heur_seq = scen.solution_by_name(start_seq_name)
    except KeyError:    
        heur_drone, heur_seq = pmu.search_heuristic_closest(scen.drone, scen.targets)
        heur_dur = heur_drone.flight_duration()
    print(f'Starting point (heuristic closest target {heur_dur})')
    pm_sa._print_sol(0, heur_dur, heur_dur, heur_seq)
    print(f'Local search ({ntest} iterations)')
    if debug:
        #best_drone, best_seq, all_durs, kept_durs, Paccept = search_locally(scen.drone, scen.targets, heur_dur, heur_seq, ntest, debug, Tf)
        best_drone, best_seq, all_durs, kept_durs, Paccept = pm_sa.search(scen.drone, scen.targets, start_dur=None, start_seq=None, ntest=ntest, debug=True, Tf=Tf, display=True)
        return best_drone, best_seq, all_durs, kept_durs, Paccept
    else:
        #best_drone, best_seq = search_locally(scen.drone, scen.targets, heur_dur, heur_seq, ntest)
        best_drone, best_seq = pm_sa.search(scen.drone, scen.targets, start_dur=None, start_seq=None, ntest=ntest, debug=False, Tf=Tf, display=False)
        return best_drone, best_seq


#
# 
#
def test1(filename = 'scenario_30_1.yaml', epochs=int(1e4)):
    scen = pmu.Scenario(filename=filename)
    try:
        scen.solution_by_name('hc')
    except KeyError:
        test_drone, test_seq = pmu.search_heuristic_closest(scen.drone, scen.targets)
        scen.add_solution('hc', test_drone.flight_duration(), test_seq)
    
    best_drone, best_seq = search_heuristic_closest_refined(scen, ntest=epochs)
    scen.add_solution('heuristic_refined', best_drone.flight_duration(), best_seq)
    scen.save(f'/tmp/{filename}')


def mmmm(_v): return np.min(_v), np.max(_v), np.mean(_v), np.median(_v)
def _fmt_mmmm(_v): _m=mmmm(_v); return f'min/max/mean/med: {_m[0]:.3f} / {_m[1]:.3f} / {_m[2]:.3f} / {_m[3]:.3f}'
def _print_summary(_i, _v): print(f'case: {_i} -> {_fmt_mmmm(_v)}')
    
def plot_search_chronograms(filename, epoch, nrun, force=False):
    scen = pmu.Scenario(filename=filename)
    # set of quasi lin
    #Ts = [lambda i: pm_sa._f1(2, epoch/10, 1e-2, 9*epoch/10, i)]*5
    # set of quasi lin
    #Ts = [lambda i: pm_sa._f1(5, epoch/10, 1e-2, 9*epoch/10, i), lambda i: pm_sa._f1(2, epoch/10, 1e-2, 9*epoch/10, i), lambda i: pm_sa._f1(1, epoch/10, 1e-2, 9*epoch/10, i)]
    #Ts = [lambda i: pm_sa._f1(2, epoch/10, 1e-2, 9*epoch/10, i), lambda i: pm_sa._f1(1, epoch/10, 1e-2, 9*epoch/10, i)]
    # lin vs quasi lin
    #Ts = [lambda i: pm_sa._aff(2, 1e-2, epoch, i), lambda i: pm_sa._f1(2, epoch/10, 1e-2, 9*epoch/10, i), lambda i: pm_sa._f1(2, 0, 1e-2, 9*epoch/10, i)]
    #Ts = [lambda i: pm_sa._aff(2, 0.01, epoch, i), lambda i: pm_sa._f1(2, epoch/10, 1e-2, 9*epoch/10, i)]
    Ts = [lambda i: pm_sa._aff(1, 1e-2, epoch, i), lambda i: pm_sa._f1(1, 0, 1e-2, 9*epoch/10, i), lambda i: pm_sa._f1(1, 0, 1e-2, 8*epoch/10, i)]
    # set of lin
    #Ts = [lambda i: pm_sa._aff(T0, 1e-2, epoch, i) for T0 in [5, 2, 1,]]
    #Ts = [lambda i: pm_sa._aff(5, 1e-2, epoch, i), lambda i: pm_sa._aff(2, 1e-2, epoch, i), lambda i: pm_sa._aff(1, 1e-2, epoch, i)]
    # step
    #Ts = [lambda i: pm_sa._aff(1, 1e-2, epoch, i), lambda i: pm_sa._step(1, 1e-2, epoch/2, i)]
    # 2 aff
    #Ts = [lambda i: pm_sa._2aff(1, 1e-2, epoch/2, 0.5, 1e-2, epoch, i)]
    #Ts = [lambda i: pm_sa._aff(1, 1e-2, epoch, i), lambda i: pm_sa._2aff(1, 1e-2, 2*epoch/3, 0.25, 1e-2, epoch, i), lambda i: pm_sa._f1(1, 0, 1e-2, 2*epoch/3, i)]
    # lin vs exp
    #Ts = [lambda i: pm_sa._aff(2, 0.01, epoch, i), lambda i: pm_sa._exp(2, epoch/2, i), lambda i: pm_sa._exp(2, epoch/3, i)]
    # default lin
    #Ts = [lambda i: pm_sa._aff(1, 1e-2, epoch, i)]

    cache_filename = f'/tmp/psc_{scen.name}_{epoch}.npz'
    if force or not os.path.exists(cache_filename):
        print(f'running optimizations and storing to {cache_filename}')
        res = []
        for _i, Tf in enumerate(Ts):
            res.append([pm_sa.search(scen.drone, scen.targets, ntest=epoch, debug=True, Tf=Tf, display=0, use_native=True) for _nr in range(nrun)])
            _print_summary(_i, [_r[0].flight_duration() for _r in res[-1]])
        np.savez(cache_filename, res=res)#, allow_pickle=True)
    else:
        print(f'loading optimization results from {cache_filename}')
        _d = np.load(cache_filename, allow_pickle=True)
        res = _d['res']

    
    costs_by_runs = [[_l[0].flight_duration() for _l in _r] for _r in res ]

    fig = plt.figure(constrained_layout=True, figsize=[10.24, 5.12])#, tight_layout=True)
    fig.canvas.set_window_title(filename)
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, :-1])
    ax2 = fig.add_subplot(gs[1, :-1])
    ax3 = fig.add_subplot(gs[2, :-1])
    #ax1, ax2, ax3 = fig.subplots(3,1, sharex=True)
    axes_chrono = [fig.add_subplot(gs[_i,-1]) for _i in range(len(Ts))]

    colors_by_cases = []
    _start = 0#int(9./10*epoch)

    for k, (Tf, _res) in enumerate(zip(Ts, res)):
        for _nr, (best_drone, best_seq, all_durs, kept_durs, Paccept) in enumerate(_res):
            if _nr==0:
                _l = ax1.plot(kept_durs[_start:], label=f'{k}', alpha=0.6)
                colors_by_cases.append(_l[0].get_color())
                ax3.plot([Ts[k](i) for i in range(_start, epoch)], label=f'{k}')
            else:
                ax1.plot(kept_durs[_start:], color=colors_by_cases[-1], alpha=0.6)
            ax2.plot(Paccept[_start:], color=colors_by_cases[-1], alpha=0.6)#, label=f'{k}')

    #pdb.set_trace()
    ylim = (0, 300)
    pmu.decorate(ax1, 'Cost', ylab='s', legend=True, ylim=ylim)
    pmu.decorate(ax2, 'Paccept')#, legend=True)
    pmu.decorate(ax3, 'Temperature', xlab='episode', ylab='s', legend=True)
    plt.figure()
    #fig = plt.gcf(); fig.canvas.set_window_title(filename)
    #axes = fig.subplots(len(Ts),1, sharex=True, sharey=True, squeeze=False)
    for _i, (_c, _ax, _co) in enumerate(zip(costs_by_runs, axes_chrono, colors_by_cases)):
        _ax.hist(_c, alpha=0.6, color=_co)
        pmu.decorate(_ax, f'Final cost case {_i}', xlab='time in s')

    
# TODO
# *sample solutions in scenario  -> 
# *fix chronogram
# *sa with start hc

import sample_random_population as srp

def main(id_scen=13):
    #test1(filename = 'scenario_30_2.yaml')
    
    #plot_search_chronograms(filename = pmu.ressource('data/scenario_30_1.yaml'), epoch=int(2e4), nrun=50, force=False)
    #plot_search_chronograms(filename = pmu.ressource('data/scenario_30_2.yaml'), epoch=int(3e4), nrun=50, force=False)
    plot_search_chronograms(filename = pmu.ressource('data/scen_30/4.yaml'), epoch=int(2e4), nrun=50, force=False)
    #plot_search_chronograms(filename = pmu.ressource('data/scen_60/9_1.yaml'), epoch=int(5e4), nrun=25, force=False)
    #sample_solutions(['../data/scenario_30_1.yaml'], nb_samples=int(1e5))
    #sample_solutions(['../data/scenario_30_2.yaml'], nb_samples=int(1e5))
    #sample_solutions(['../data/scenario_30_4.yaml', '../data/scenario_30_6.yaml', '../data/scenario_30_9.yaml'], nb_samples=int(1e5))


    def _sample_for_size(size, ids, nb_samples):
        filenames = [pmu.ressource(f'data/scen_{size}/{_id}.yaml') for _id in ids]
        srp.sample_solutions(filenames, nb_samples)
        
    #_sample_for_size(15, [1, 2, 3, 4, 6, 9], int(1e5))
    #_sample_for_size(30, [1, 2, 3, 4, 6, 9], int(1e6))
    #_sample_for_size(60, [1, 2, 3, 4, 6, 9], int(1e5))
    #_sample_for_size(120, [1, 2, 3, 4, 6, 9], int(1e5))
    #_sample_for_size(240, [1, 4, 6, 9], int(1e5)) # fails for scen 2 and 3
    #_sample_for_size(480, [4, 6, 9], int(1e5)) # fails for scen 1, 2 and 3
    #_sample_for_size(960, [4, 6, 9], int(1e5)) # fails for scen 1, 2 and 3
    
    plt.show()
    
if __name__ == '__main__':
    main()
    
