#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Unit test for heuristics
'''
import numpy as np, matplotlib.pyplot as plt
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

def anim_side_by_side(_d1, _t1, _d2, _t2, titles):
    fig = plt.figure(tight_layout=True, figsize=[10.24, 5.12])
    ax1, ax2 = fig.subplots(1,2, sharex=True)
    ax1.axis('equal')
    dur = max(_d1.flight_duration(), _d2.flight_duration())
    anim1 = pma.animate(fig, ax1, _d1, _t1, t0=0., t1=dur, dt=0.1, xlim=(-100, 200), ylim=(-150, 150), title=titles[0])
    anim2 = pma.animate(fig, ax2, _d2, _t2, t0=0., t1=dur, dt=0.1, xlim=(-100, 200), ylim=(-150, 150), title=titles[1])
    #pma.save_animation(anim1, '/tmp/anim.mp4', dt=0.1)
    

def test_heuristic():
    drone, targets = pmu.make_conv_div_scenario(ntarg=30, dp0=(30,5), dv=15, tv_mean=5., tv_std=1.5, conv=True)
    #drone, targets = pmu.make_random_scenario(ntarg=20, dp0=(10,0), dv=15)
    test_drone, test_target = search_heuristic_closest(drone, targets)
    print(f'{len(targets)} targets, heuristic {test_drone.flight_duration():.1f} s')
    anim = pma.animate(plt.gcf(), plt.gca(), test_drone, test_target,
                       t0=0., t1=test_drone.ts[-1], dt=0.1, xlim=(-100, 200), ylim=(-150, 150))
    plt.show()
    pma.save_animation(anim, '/tmp/anim.mp4', dt=0.1)

def compare_with_optimal():
    #drone, targets = pmu.load_scenario('./scenario_6.yaml')
    #drone, targets = pmu.make_random_scenario(ntarg=6, dp0=(10,0), dv=15)
    #drone, targets = pmu.make_conv_div_scenario(ntarg=7, dp0=(30,5), dv=15, tv_mean=5., tv_std=1.5, conv=True)
    drone, targets = pmu.make_conv_div_scenario(ntarg=8, dp0=(30,5), dv=15, tv_mean=5., tv_std=1.5, conv=False)
    print(f'Exhaustive search for {len(targets)} targets')
    best_drone, best_target = pmu.search_exhaustive(drone, targets)
    test_drone, test_target = search_heuristic_closest(drone, targets)
    print(f'optimal {best_drone.flight_duration():.1f} s heuristic {test_drone.flight_duration():.1f} s')
    anim_side_by_side(best_drone, best_target, test_drone, test_target, ['optimal', 'heuristic'])
 
def main():
    #compare_with_optimal()
    test_heuristic()
    plt.show()
    
if __name__ == '__main__':
    main()
