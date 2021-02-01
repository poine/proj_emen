#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Runs search on a scenario
'''

import sys, argparse

import proj_manen as pm, proj_manen_utils as pmu, proj_manen.simulated_annealing as pm_sa

def main(filename, method='exhaustive', max_epoch=10000, save_sol_name=None, save_filename=None):
    scen = pmu.Scenario(filename=filename)
    if method == 'exhaustive':
        _drone, _seq = pm.search_exhaustive(scen.drone, scen.targets, keep_all=False)
    elif method == 'heuristic_closest':
        _drone, _seq = pm.search_heuristic_closest(scen.drone, scen.targets)
    elif method == 'sa':
        _drone, _seq = pm_sa.search(scen.drone, scen.targets, ntest=max_epoch)
    
    print(f'{_drone.flight_duration():.2f}')

    if save_filename is None:
        scen.add_solution(save_sol_name, _drone.flight_duration(), _seq)
        scen.save(save_filename)
    
        
if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='Runs search on a scenario.')
     parser.add_argument("filename")
     parser.add_argument('-m', '--method', default='exhaustive', help='search method: exhaustive, heuristic_closest, sa')
     parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epoch for sa')
     parser.add_argument('-s', '--save_filename', help='save scenario to file')
     parser.add_argument('-n', '--sol_name', help='name of solution (for saving)')
     args = parser.parse_args()
     main(args.filename, args.method, args.epoch, args.sol_name, args.save_filename)
