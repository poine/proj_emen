#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Unit test for heuristics
'''
import os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.animations as pma

# def search_heuristic_closest(drone, targets):
#     drone = copy.deepcopy(drone) # we don't change our input arguments
#     remaining, solution = targets.copy(), []
#     while len(remaining) > 0:
#         now = drone.flight_duration()
#         rel_tpos = [_targ.get_pos(now)-drone.get_pos(now) for _targ in remaining]
#         closest_target = remaining[np.argmin(np.linalg.norm(rel_tpos, axis=1))]
#         remaining.remove(closest_target);solution.append(closest_target)
#         psi, dt = pm.intercept_1(drone, closest_target)
#         drone.add_leg(dt, psi)
#     return drone, solution

def search_exhaustive_threshold(drone, targets):
    test_drone, test_seq = search_heuristic_closest(drone, targets)
    best_drone, best_seq = pm.search_exhaustive_with_threshold(drone, targets, test_drone.flight_duration())
    return  best_drone, best_seq

def test_heuristic(drone, targets):
    _start = time.perf_counter()
    test_drone, test_target = search_heuristic_closest(drone, targets)
    _end = time.perf_counter()
    print(f'{len(targets)} targets, heuristic {test_drone.flight_duration():.1f} s')
    anim = pma.animate(plt.gcf(), plt.gca(), test_drone, test_target,
                       t0=0., t1=test_drone.ts[-1], dt=0.1, xlim=(-100, 200), ylim=(-150, 150))
    return anim
    
def compare_with_optimal(scen):
    try:
        name, opt_dur, opt_seq = scen.solution_by_name('optimal')
        opt_drone, _ = pm.intercept_sequence_copy(scen.drone, opt_seq)
    except KeyError:    
        opt_drone, opt_seq = pm.search_exhaustive(scen.drone, scen.targets)
        opt_dur = opt_drone.flight_duration()
        
    try:
        name, heur_dur, heur_seq = scen.solution_by_name('heuristic')
        heur_drone, _ = pm.intercept_sequence_copy(scen.drone, heur_seq)
    except KeyError:    
        heur_drone, heur_seq = pm_t3.search_heuristic_closest(scen.drone, scen.targets)
        heur_dur = heur_drone.flight_duration()

    print(f'optimal {opt_dur:.1f} s heuristic {heur_dur:.1f} s')

    fig = plt.figure(tight_layout=True, figsize=[10.24, 5.12])
    ax1, ax2 = fig.subplots(1,2, sharex=True)
    return pma.animate_multi(fig, [ax1, ax2], [opt_drone, heur_drone], [opt_seq, heur_seq], ['Optimal', 'Heuristic'])

# moved to animation module
# def animate_solutions(scen, names):

#     sols = [scen.solution_by_name(name) for name in names]
#     seqs = [_seq for _1, _2, _seq in sols]
#     drones = [pm.intercept_sequence_copy(scen.drone, _seq)[0] for _seq in seqs] 

#     _n = len(names)
#     fig = plt.figure(tight_layout=True, figsize=[5.12*_n, 5.12])
#     axes = fig.subplots(1,_n, sharex=True)
#     if _n == 1: axes=[axes]
#     return pma.animate_multi(fig, axes, drones, seqs, names)
        


def plot_histogram(fig, ax, drone, targets, title):
    drones, targetss = pm.search_exhaustive(drone, targets, keep_all=True)
    test_drone, test_target = search_heuristic_closest(drone, targets)
    _cs = np.array([_d.flight_duration() for _d in drones])
    _ib, _iw = np.argmin(_cs), np.argmax(_cs)
    _cb = drones[_ib].flight_duration()
    _ct = test_drone.flight_duration()
    print(f'optimal {_cb:.1f} s heuristic {_ct:.1f} s')
    _n, _b, _p = ax.hist(_cs, bins=30)
    ax.plot([_ct, _ct], [0, 300], label=f'heuristic ({_ct:.2f} s)')
    ax.plot([_cb, _cb], [0, 300], label=f'optimal ({_cb:.2f} s)')
    pmu.decorate(ax, title, 'time in s', 'nb sequences', True)

def plot_one_histogram(drone, targets, title=''):
    plot_histogram(plt.gcf(), plt.gca(), drone, targets, title)
    
def plot_scens_histograms():
    _scens = [make_or_load_scenario(_i) for _i in [0, 1, 2, 3]]
    #_scens = [make_or_load_scenario(_i) for _i in [4, 5, 6, 7]]
    _titles = [f'scenario {_i}' for _i in range(len(_scens))]
    fig = plt.figure(tight_layout=True, figsize=[10.24, 5.12])
    axes = fig.subplots(1,len(_scens))#, sharex=True)
    for (_d, _t, _s), _title, _ax in zip(_scens, _titles, axes):
        plot_histogram(fig, _ax, _d, _t, _title)

def test_heuristic_3(drone, targets):
    _start = time.perf_counter()
    drones, targetss = pm.search_exhaustive(drone, targets, keep_all=True)
    _end = time.perf_counter()
    print(f'Exhaustive search took {_end-_start:.1f} s')
    _start = time.perf_counter()
    test_drone, test_target = search_heuristic_closest(drone, targets)
    _foo, _bar = pm.search_exhaustive_with_threshold(drone, targets, test_drone.flight_duration())
    _end = time.perf_counter()
    print(f'Exhaustive thresholded search took {_end-_start:.1f} s')


def anim_scens(show_opt=True, show_heu=True, show_hist=True, _fs=3.84):
    _scens = [make_or_load_scenario(_i) for _i in [0, 1, 2, 3]]
    #_scens = [make_or_load_scenario(_i) for _i in [4, 5, 6, 7]]
    #_scens = [make_or_load_scenario(_i) for _i in [8, 9]]
    #_scens = [make_or_load_scenario(_i) for _i in [10, 11]]
    _sols = [_3 for _1, _2, _3 in _scens]
    drones, targets = [],[]
    if show_opt:
        for _d, _t, _s in _scens:
            best_dur, best_seq = pmu.sol_by_name(_s, 'optimal')
            if best_seq is not None:
                best_drone, best_dur = pm.intercept_sequence_copy(_d, best_seq)
            else:
                best_drone, best_seq = pm.search_exhaustive(_d, _t)
            drones.append(best_drone), targets.append(best_seq)
    if show_heu:
        for _d, _t, _s in _scens:
            test_dur, test_seq = pmu.sol_by_name(_s, 'heuristic')
            if test_seq is not None: test_drone, test_dur = pm.intercept_sequence_copy(_d, test_seq)
            else: test_drone, test_seq = search_heuristic_closest(_d, _t)
            drones.append(test_drone), targets.append(test_seq)

    _nr, _nc = np.sum([show_opt, show_heu, show_hist]), len(_scens)
    titles = [f'scenario {_i}_opt' for _i in range(_nc)]+[f'scenario {_i}_heur' for _i in range(_nc)]
    fig = plt.figure(tight_layout=True, figsize=[_fs*_nc, _fs*_nr])
    axes = fig.subplots(_nr, _nc)#, sharex=True)
    if show_hist:
        _titles = [f'scenario {_i} histogram' for _i in range(_nc)]
        for (_d, _t, _s), _title, _ax in zip(_scens, _titles, axes[2,:]):
            plot_histogram(fig, _ax, _d, _t, _title)

    return pma.animate_multi(fig, axes.flatten(), drones, targets, titles, xlim=(-150, 150), ylim=(-150, 150))


def save_solutions(idx):
    drone, targets, solutions = make_or_load_scenario(idx)
    test_drone, test_seq = search_heuristic_closest(drone, targets)
    #best_drone, best_seq = pm.search_exhaustive(drone, targets)
    best_drone, best_seq = pm.search_exhaustive_with_threshold(drone, targets, test_drone.flight_duration())
    print(f'optimal {best_drone.flight_duration():.1f} s heuristic {test_drone.flight_duration():.1f} s')
    #pdb.set_trace()
    def _format_sol(_n, _d, _s): return [_n, _d.flight_duration(), [_t.name for _t in _s]]
    solutions = [_format_sol('optimal', best_drone, best_seq), _format_sol('heuristic', test_drone, test_seq)]
    pmu.save_scenario('/tmp/foo.yaml', drone, targets, solutions)


def add_solution(idx, drone, sequence, name):
    _drone, _targets, _solutions = make_or_load_scenario(idx)
    
    
    
def plot_cpu_time():
    ntargs, dts, dts2 = [2, 3, 4, 5, 6, 7, 8], [], []
    for ntarg in ntargs:
        drone, targets = pmu.make_random_scenario(ntarg=ntarg, dp0=(0,0), dv=15)
        _start = time.perf_counter()
        best_drone, best_seq = search_exhaustive_threshold(drone, targets)
        _end = time.perf_counter(); dts2.append(_end-_start)
        print(f'{ntarg} targets thresh-> {dts2[-1]:.1f} s')
        _start = time.perf_counter()
        best_drone, best_seq = pm.search_exhaustive(drone, targets)
        _end = time.perf_counter(); dts.append(_end-_start)
        print(f'{ntarg} targets -> {dts[-1]:.1f} s')
    plt.plot(ntargs, dts, '--o', label='Exhaustive search')
    plt.plot(ntargs, dts2, '--o', label='Exhaustive search with threshold')
    pmu.decorate(plt.gca(), 'cpu time', 'number of targets', 'time in s', True)
    plt.savefig('../docs/images/ex_search_time_vs_size_2.png')
    
def main():
    #plot_cpu_time()
    #save_solutions(1)
    #drone, targets, solutions = make_or_load_scenario(1, make=False)
    scen = load_scenario(1)
    #anim = test_heuristic(drone, targets)
    #anim = compare_with_optimal(scen)
    anim = animate_solutions(scen, ['optimal', 'heuristic'])
    #plot_one_histogram(drone, targets)
    #plot_scens_histograms()
    #test_heuristic_3(drone, targets)
    #anim = anim_scens()
    #plt.savefig('../docs/images/histo_scens_7.png')
    plt.show()
    #pma.save_animation(anim, '/tmp/anim.mp4', dt=0.1)
    
if __name__ == '__main__':
    main()
