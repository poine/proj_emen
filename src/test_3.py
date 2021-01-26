#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Unit test for heuristics
'''
import os, time, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen_utils as pmu, animations as pma

def search_heuristic_closest(drone, targets):
    remaining, solution = targets.copy(), []
    while len(remaining) > 0:
        now = drone.flight_duration()
        rel_tpos = [_targ.get_pos(now)-drone.get_pos(now) for _targ in remaining]
        target = remaining[np.argmin(np.linalg.norm(rel_tpos, axis=1))]
        remaining.remove(target);solution.append(target)
        psi, dt = pm.intercept_1(drone, target)
        drone.add_leg(dt, psi)
    return drone, solution

# works but won't allow to save :(
def anim_side_by_side(_d1, _t1, _d2, _t2, titles):
    fig = plt.figure(tight_layout=True, figsize=[10.24, 5.12])
    ax1, ax2 = fig.subplots(1,2, sharex=True)
    ax1.axis('equal')
    dur = max(_d1.flight_duration(), _d2.flight_duration())
    anim1 = pma.animate(fig, ax1, _d1, _t1, t0=0., t1=dur, dt=0.1, xlim=(-100, 200), ylim=(-150, 150), title=titles[0])
    anim2 = pma.animate(fig, ax2, _d2, _t2, t0=0., t1=dur, dt=0.1, xlim=(-100, 200), ylim=(-150, 150), title=titles[1])
    #pma.save_animation(anim1, '/tmp/anim.mp4', dt=0.1)
    

def test_heuristic(drone, targets):
    _start = time.perf_counter()
    test_drone, test_target = search_heuristic_closest(drone, targets)
    _end = time.perf_counter()
    print(f'{len(targets)} targets, heuristic {test_drone.flight_duration():.1f} s')
    anim = pma.animate(plt.gcf(), plt.gca(), test_drone, test_target,
                       t0=0., t1=test_drone.ts[-1], dt=0.1, xlim=(-100, 200), ylim=(-150, 150))
    return anim
    
def compare_with_optimal(drone, targets):
    print(f'Exhaustive search for {len(targets)} targets')
    best_drone, best_target = pm.search_exhaustive(drone, targets)
    test_drone, test_target = search_heuristic_closest(drone, targets)
    print(f'optimal {best_drone.flight_duration():.1f} s heuristic {test_drone.flight_duration():.1f} s')
    #anim_side_by_side(best_drone, best_target, test_drone, test_target, ['optimal', 'heuristic'])
    fig = plt.figure(tight_layout=True, figsize=[10.24, 5.12])
    ax1, ax2 = fig.subplots(1,2, sharex=True)
    return pma.animate_multi(fig, [ax1, ax2], [best_drone, test_drone], [best_target, test_target], ['Optimal', 'Heuristic'])
    
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
    plot_histogram(plt.gcf(), plt.gca(), _d, _t, _name)
    
def plot_scens_histograms():
    _scens = [make_or_load_scenario(_i) for _i in [0, 1, 2, 3]]
    #_scens = [make_or_load_scenario(_i) for _i in [4, 5, 6, 7]]
    _titles = [f'scenario {_i}' for _i in range(len(_scens))]
    fig = plt.figure(tight_layout=True, figsize=[10.24, 5.12])
    axes = fig.subplots(1,len(_scens))#, sharex=True)
    for (_d, _t), _title, _ax in zip(_scens, _titles, axes):
        plot_histogram(fig, _ax, _d, _t, _title)
    

def test_heuristic_3(drone, targets):
    _start = time.perf_counter()
    drones, targetss = pm.search_exhaustive(drone, targets, keep_all=True)
    _end = time.perf_counter()
    print(f'Exhaustive search took {_end-_start:.1f} s')
    _start = time.perf_counter()
    test_drone, test_target = search_heuristic_closest(drone, targets)
    _end = time.perf_counter()
    print(f'Heuristic took {_end-_start:.1f} s')
    _start = time.perf_counter()
    _foo, _bar = pm.search_exhaustive_with_threshold(drone, targets, test_drone.flight_duration())
    _end = time.perf_counter()
    print(f'Exhaustive thresholded search took {_end-_start:.1f} s')


def anim_scens(show_opt=False):
    #_scens = [make_or_load_scenario(_i) for _i in [0, 1, 2, 3]]
    _scens = [make_or_load_scenario(_i) for _i in [4, 5, 6, 7]]
    #_scens = [make_or_load_scenario(_i) for _i in [8, 9]]
    #_scens = [make_or_load_scenario(_i) for _i in [10, 11]]

    if show_opt:
        _opts = [pm.search_exhaustive(_d, _t) for _d, _t in _scens]
        drones, targets = [_1 for _1, _2 in _opts], [_2 for _1, _2 in _opts]
    else:
        _heurs = [search_heuristic_closest(_d, _t) for _d, _t in _scens]
        drones, targets = [_1 for _1, _2 in _heurs], [_2 for _1, _2 in _heurs]

    titles = [f'scenario {_i}' for _i in range(len(_scens))]
    fig = plt.figure(tight_layout=True, figsize=[5.12*len(_scens), 5.12])
    axes = fig.subplots(1,len(_scens), sharex=True)

    return pma.animate_multi(fig, axes, drones, targets, titles)

def make_or_load_scenario(idx, make=False):
    filenames = ['scenario_7_1.yaml', 'scenario_7_2.yaml', 'scenario_7_3.yaml', 'scenario_7_4.yaml',
                 'scenario_8_1.yaml', 'scenario_8_2.yaml', 'scenario_8_3.yaml', 'scenario_8_4.yaml',
                 'scenario_30_1.yaml', 'scenario_30_2.yaml', 'scenario_60_1.yaml', 'scenario_60_2.yaml']
    if make or not os.path.exists(filenames[idx]):
        if   idx == 0: drone, targets = pmu.make_random_scenario(ntarg=7, dp0=(10,0), dv=15)
        elif idx == 1: drone, targets = pmu.make_conv_div_scenario(ntarg=7, dp0=(30,5), dv=15, tv_mean=5., tv_std=1.5, conv=True)
        elif idx == 2: drone, targets = pmu.make_conv_div_scenario(ntarg=7, dp0=(30,5), dv=15, tv_mean=5., tv_std=1.5, conv=False)
        elif idx == 3: drone, targets = pmu.make_conv_div_scenario(ntarg=7, dp0=(30,5), dv=15, tv_mean=6., tv_std=1.5, conv=False)
        elif idx == 4: drone, targets = pmu.make_random_scenario(ntarg=8, dp0=(10,0), dv=15)
        elif idx == 5: drone, targets = pmu.make_conv_div_scenario(ntarg=8, dp0=(30,5), dv=15, tv_mean=5., tv_std=1.5, conv=True)
        elif idx == 6: drone, targets = pmu.make_conv_div_scenario(ntarg=8, dp0=(30,5), dv=15, tv_mean=5., tv_std=1.5, conv=False)
        elif idx == 7: drone, targets = pmu.make_random_scenario(ntarg=8, dp0=(10,0), dv=15)
        elif idx == 8: drone, targets = pmu.make_conv_div_scenario(ntarg=30, dp0=(30,5), dv=15, tv_mean=5., tv_std=1.5, conv=True)
        elif idx == 9: drone, targets = pmu.make_conv_div_scenario(ntarg=30, dp0=(30,5), dv=15, tv_mean=5., tv_std=1.5, conv=False)
        elif idx == 10: drone, targets = pmu.make_conv_div_scenario(ntarg=60, dp0=(30,5), dv=15, tv_mean=5., tv_std=1.5, conv=True)
        elif idx == 11: drone, targets = pmu.make_conv_div_scenario(ntarg=60, dp0=(30,5), dv=15, tv_mean=5., tv_std=1.5, conv=False)
        pmu.save_scenario(filenames[idx], drone, targets)
    else:
        drone, targets = pmu.load_scenario(filenames[idx])
    return drone, targets
    

def main():
    #drone, targets = make_or_load_scenario(0, make=False)
    #anim = test_heuristic(drone, targets)
    #anim = compare_with_optimal(drone, targets)
    #anim = plot_histogram(drone, targets)
    plot_scens_histograms()
    #test_heuristic_3(drone, targets)
    #anim = anim_scens()
    #plt.savefig('../docs/images/histo_scens_7.png')
    plt.show()
    #pma.save_animation(anim, '/tmp/anim.mp4', dt=0.1)
    
if __name__ == '__main__':
    main()
