#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Unit test multi drone case
'''
import argparse, sys, os, time, copy, numpy as np, matplotlib.pyplot as plt
import multiprocessing as mp
import pdb

import proj_manen as pm, proj_manen.utils as pmu
import proj_manen.scenarios as pm_sc

import proj_manen.multi_drone as pm_md
import proj_manen.native_core as pm_nc

    

def main(filename, method='sa3', epochs=int(1e5), sol_name='latest', T0=2., save=False):
    scen = pm_sc.Scenario(filename=filename)
    if method == 'sa': # Python
        _drones, _seqs = pm_md.search(scen.drones, scen.targets, epochs, T0=T0, display=2)
    elif method == 'sa2': # Python/C++
        _drones, _seqs = pm_md.search(scen.drones, scen.targets, epochs, T0=T0, display=2, use_native=True)
    elif method == 'sa3': # C++
        _drones, _seqs = pm_nc.search_sa_md(scen.drones, scen.targets, epochs, T0=T0, display=2)
        
    if save:
        cost = np.max([_d.flight_duration() for _d in _drones])
        scen.add_solution(sol_name, cost, _seqs)
        save_filename = filename#'/tmp/foo.yaml'
        scen.save(save_filename) 
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs search on a scenario.')
    parser.add_argument("filename")
    parser.add_argument('-m', '--method', default='sa3', help='search method: sa, sa3')
    parser.add_argument('-e', '--epoch', help='number of epoch for sa', default='1k')
    parser.add_argument('-w', '--write', help='save solution in original file', action="store_true")
    parser.add_argument('-f', '--T0', type=float, default=2., help='start temperature for sa')
    parser.add_argument('-n', '--sol_name', default='latest', help='name of solution')
    args = parser.parse_args()
    main(args.filename, args.method, pmu.parse_with_prefix(args.epoch),  args.sol_name, args.T0, args.write)

