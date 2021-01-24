#! /usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np, scipy.optimize, matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from itertools import permutations
import pdb

import proj_manen as pm


def plot(ax, _d, _targets, plot_targets=True):
    poses = np.asarray(_d.Xs)
    ax.plot(poses[:,0], poses[:,1], 'o', label=f'drone {_d.v:.1f} m/s')
    for _p, _psi, _t in zip(_d.Xs, _d.psis, _d.ts[1:]):
        vx, vy = pm._to_eucl(_d.v, _psi)
        ax.arrow(_p[0], _p[1], vx, vy, width=0.15, head_width=_d.v*0.05, head_length=_d.v*0.1, length_includes_head=True)
        _p1 = _d.get_pos(_t)
        ax.plot((_p[0], _p1[0]), (_p[1], _p1[1]), '--', color='C0')

    if plot_targets:
        for _targ in _targets:
            pc0 = _targ.get_pos(0)
            ax.plot(pc0[0], pc0[1], 'o', color='C1', label=f'{_targ.name} ${np.rad2deg(_targ.psi):.1f} deg {_targ.v:.1f} m/s$')
            ax.arrow(pc0[0], pc0[1], _targ.vx, _targ.vy, width=0.2, head_width=_targ.v*0.05, head_length=_targ.v*0.1, length_includes_head=True)

        for _ts, _targ in zip(_d.ts[1:], _targets):
            pi = _targ.get_pos(_ts)
            p0 = _targ.get_pos(0)
            ax.plot(pi[0], pi[1], 'o', color='C2', label=f'interception {_targ.name} {_ts:.2f} s')
            ax.plot((p0[0], pi[0]), (p0[1], pi[1]), '--', color='C1')
            
    ax.axis('equal');ax.legend();ax.grid()

def plot_all_sols(drone, targets, _nc=3):
    perms = set(permutations(targets))
    fig = plt.figure(tight_layout=True, figsize=[12.8, 9.6]); fig.canvas.set_window_title('Interceptions')
    _nc = min(_nc, len(perms))
    _nr, _r = np.divmod(len(perms), _nc)
    if _r>0: _nr+=1 
    axes = fig.subplots(_nr,_nc, sharex=True)#, sharey=True)
    print(f'{len(perms)} permutation')
    for targets, ax in zip(perms, axes.flatten()):
        pm.solve_sequence(drone, targets)
        plot(ax, drone, targets)
        drone.clear_traj()

def test_1(filename, rnd=False): # first solution for a given scenario
    drone, targets = pm.load_scenario(filename)
    if rnd:
        _p = rng.permutation(len(targets))
        targets = np.array(targets)[_p]
    pm.solve_sequence(drone, targets)
    plot(plt.gca(), drone, targets)

def test_2(): # plot all solutions for a 2 targets scenario
    drone, targets = pm.load_scenario('./scenario_2.yaml')
    plot_all_sols(drone, targets)
    plt.savefig('all_sols_scen2.png')

def test_3(): # plot all solutions for a 3 targets scenario
    drone, targets = pm.load_scenario('./scenario_3.yaml')
    plot_all_sols(drone, targets)
    plt.savefig('all_sols_scen3.png')

def test_4(): # plot all solutions for a 4 targets scenario
    drone, targets = pm.load_scenario('./scenario_4.yaml')
    plot_all_sols(drone, targets, _nc=4)
    plt.savefig('all_sols_scen4.png')

from copy import copy, deepcopy
def test_5(filename): # compute all solutions, keeps best
    drone, targets = pm.load_scenario(filename)
    #drone, targets = pm.make_scenario(ntarg=8, dp0=(0,0), dv=15)
    perms = set(permutations(targets))
    durations = []
    best_dur, best_targets, best_drone = float('inf'), None, None
    worst_dur, worst_targets, worst_drone = 0., None, None
    for targets in perms:
        _drone = deepcopy(drone)
        dur = pm.solve_sequence(_drone, targets)
        durations.append(dur)
        if dur < best_dur:
            best_dur, best_targets, best_drone = dur, targets, _drone
        if dur > worst_dur:
            worst_dur, worst_targets, worst_drone = dur, targets, _drone
    #best_id = np.argmin(durations)
    #print(best_id, durations[best_id])
    
    plot(plt.gca(), best_drone, best_targets)
    plt.figure()
    plot(plt.gca(), worst_drone, worst_targets)
    
    #plt.hist(durations)
    plt.show()
    
    
#test_1('./scenario_1.yaml')
test_1('./scenario_6.yaml')
#plt.savefig('one_sols_scen6.png')
#test_2()
#test_3()
#test_4()
#test_5('./scenario_6.yaml')

plt.show()
