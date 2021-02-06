#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Runs search on a scenario
'''

import sys, argparse, time, datetime, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.simulated_annealing as pm_sa

def main(filename, method='ex', max_epoch=10000, sol_name=None, save_filename=None, overwrite=False, show=False):
    scen = pmu.Scenario(filename=filename)
    _start = time.perf_counter()
    if method == 'ex':    # exhaustive
        _drone, _seq = pmu.search_exhaustive(scen.drone, scen.targets, keep_all=False, display=True)
    elif method == 'he':  # heuristic_closest
        _drone, _seq = pm.search_heuristic_closest(scen.drone, scen.targets)
    elif method == 'ex2': # exhaustive improved
        _drone, _seq = pmu.search_exhaustive_improved(scen.drone, scen.targets, display=True)
    elif method == 'ex3': # exhaustive improved2
        _drone, _seq = pmu.search_exhaustive_improved2(scen.drone, scen.targets, display=True)
    elif method == 'sa':  # simulated annealing
        _drone, _seq = pm_sa.search(scen.drone, scen.targets, ntest=max_epoch, display=2, T0=5.)
    _end = time.perf_counter()
    print(f'{sol_name}: {_drone.flight_duration():.2f}s (computed in {datetime.timedelta(seconds=_end-_start)} h:m:s)')
    
    # check
    _drone2, _dur2 = pm.intercept_sequence_copy(scen.drone, _seq)
    if _dur2 != _drone2.flight_duration() or _dur2 != _drone.flight_duration():
        print('#### FIXME ####')
        pdb.set_trace()
    
    scen.add_solution(sol_name, _drone.flight_duration(), _seq)
    if save_filename is not None:
        scen.save(save_filename)
    if overwrite:
        scen.save(filename)
    if show:
        pmu.plot_solutions(scen, [sol_name], filename)
        plt.show()
        
if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Runs search on a scenario.')
     parser.add_argument("filename")
     parser.add_argument('-m', '--method', default='ex', help='search method: ex(haustive), he(uristic_closest), ex2, sa')
     parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epoch for sa')
     parser.add_argument('-s', '--save_filename', help='save scenario to file')
     parser.add_argument('-n', '--sol_name', help='name of solution')
     parser.add_argument('-x', '--show', help='display solution', action="store_true")
     parser.add_argument('-w', '--overwrite', help='save solution in original file', action="store_true")
     args = parser.parse_args()
     main(args.filename, args.method, args.epoch, args.sol_name, args.save_filename, args.overwrite, args.show)
