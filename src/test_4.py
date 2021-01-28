#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Unit test for simulated annealing
'''
import os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen_utils as pmu, animations as pma, test_3 as pm_t3

def _names_of_seq(_s): return [_t.name.split('_')[-1] for _t in _s]
def _print_sol(_d, _s): print(f'{_d:6.2f} {_names_of_seq(_s)}')
def _mutate(_seq):
    _seq2 = _seq.copy(); i1, i2 = np.random.randint(0, high=len(_seq), size=2)
    _foo = _seq2.pop(i1); _seq2.insert(i2, _foo)
    return _seq2

def search_locally(drone, targets, best_dur, best_seq, ntest):
    for i in range(ntest):
        _s2 = _mutate(best_seq)
        _d2, _dur = pm.intercept_sequence_copy(drone, _s2)
        if _dur < best_dur:
            best_dur, best_seq = _dur, _s2
            _print_sol(best_dur, best_seq)
    return best_dur, best_seq

def search_heuristic_closest_refined(drone, targets, solutions, ntest):
    test_drone, test_seq = pm_t3.search_heuristic_closest(drone, targets)
    print('heuristic closest target')
    _print_sol(test_drone.flight_duration(), test_seq)
    print('local search')
    best_dur, best_seq = search_locally(drone, targets, test_drone.flight_duration(), test_seq, ntest)
    opt_dur, opt_seq = pmu.sol_by_name(solutions, 'optimal')
    if opt_dur is not None:
        print('optimal')
        _print_sol(opt_dur, opt_seq)

def main():
    drone, targets, solutions = pm_t3.make_or_load_scenario(11)
    search_heuristic_closest_refined(drone, targets, solutions, ntest=10000)
    
if __name__ == '__main__':
    main()
