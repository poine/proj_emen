#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Runs search on a scenario
'''

import sys, argparse, time, datetime, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.simulated_annealing as pm_sa

import proj_manen.native_core as pm_nc


def main(filename, method='sa2', max_epoch=10000, sol_name=None, save_filename=None, overwrite=False, show=False, T0=1., start_sol_name=None):
    scen = pmu.Scenario(filename=filename)
    _start = time.perf_counter()
    _neval = max_epoch
    if method == 'ex':    # exhaustive
        _neval = np.math.factorial(len(scen.targets))
        _drone, _seq = pmu.search_exhaustive(scen.drone, scen.targets, keep_all=False, display=True)
    elif method == 'ex2': # exhaustive native
        _neval = np.math.factorial(len(scen.targets))
        _drone, _seq = pm_nc.search_exhaustive(scen.drone, scen.targets)
    elif method == 'he':  # heuristic_closest
        _drone, _seq = pmu.search_heuristic_closest(scen.drone, scen.targets)
    elif method == 'sa':  # simulated annealing
        start_seq = scen.solution_by_name(start_sol_name)[2] if start_sol_name is not None else None
        _drone, _seq = pm_sa.search(scen.drone, scen.targets, epochs=max_epoch, display=2, T0=T0, use_native=False, start_seq=start_seq)
    elif method == 'sa2':  # simulated annealing native
        start_seq = scen.solution_by_name(start_sol_name)[2] if start_sol_name is not None else None
        _drone, _seq = pm_sa.search(scen.drone, scen.targets, epochs=max_epoch, display=2, T0=T0, use_native=True, start_seq=start_seq)
    else: print('unknown search method')
    _end = time.perf_counter()
    cpu_dur = _end-_start; eval_per_sec = _neval/cpu_dur
    print(f'{sol_name}: {_drone.flight_duration():.2f}s (computed in {datetime.timedelta(seconds=cpu_dur)} h:m:s, {eval_per_sec:.0f} ev/s)')
    
    # check
    _drone2, _dur2 = pm.intercept_sequence_copy(scen.drone, _seq)
    if _dur2 != _drone2.flight_duration() or\
        not np.allclose([_dur2], [_drone.flight_duration()]):  # python and C did not recompute same duration for sequence 
        print('#### search.py: check failed FIXME ####')
        print(f'{_dur2} {_drone.flight_duration()}')
        pdb.set_trace()
    
    scen.add_solution(sol_name, _drone.flight_duration(), _seq)
    if save_filename is not None:
        scen.save(save_filename)
    if overwrite:
        scen.save(filename)
    if show:
        pmu.plot_solutions(scen, [sol_name], filename)
        
if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Runs search on a scenario.')
     parser.add_argument("filename")
     parser.add_argument('-m', '--method', default='sa', help='search method: ex(haustive), he(uristic_closest), ex2, sa')
     parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epoch for sa')
     parser.add_argument('-f', '--T0', type=float, default=1., help='start temperature for sa')
     parser.add_argument('-g', '--s0', type=str, help='name of start solution for sa')
     parser.add_argument('-s', '--save_filename', help='save scenario to file')
     parser.add_argument('-n', '--sol_name', help='name of solution')
     parser.add_argument('-x', '--show', help='display solution', action="store_true")
     parser.add_argument('-w', '--overwrite', help='save solution in original file', action="store_true")
     args = parser.parse_args()
     main(args.filename, args.method, args.epoch, args.sol_name, args.save_filename, args.overwrite, args.show, args.T0, args.s0)
     plt.show()
