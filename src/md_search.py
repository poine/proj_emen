#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Unit test multi drone case
'''
import argparse, sys, os, time, copy, numpy as np, matplotlib.pyplot as plt
import multiprocessing as mp
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.simulated_annealing as pm_sa
import proj_manen.scenarios as pm_sc

import proj_manen.multi_drone as pm_md

import proj_manen.native_core as pm_nc
try:
    import pm_cpp_ext
except ImportError:
    print('proj_manen.simulated_annealing: failed to import native code')
    

    

def main(filename='../data/scen_30/6.yaml', epochs=int(1e5), T0=2., save=False):
    scen = pm_sc.Scenario(filename=filename)
    # HACK
    #scen.drones = [scen.drone, pm.Drone((0,0), 15., np.deg2rad(0.)), pm.Drone((0,0), 15., np.deg2rad(0.))]
    #for i, _d in enumerate(scen.drones): _d.name=i
    #drone, seq = pm_sa.search(scen.drone, scen.targets, start_seq=None, epochs=int(1e4), display=1, use_native=True)
    _drones, _seqs = pm_md.search(scen.drones, scen.targets, epochs, T0=T0, display=2)
    if save:
        cost = np.max([_d.flight_duration() for _d in _drones])
        scen.add_solution('foo', cost, _seqs)
        save_filename = filename#'/tmp/foo.yaml'
        scen.save(save_filename) 
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs search on a scenario.')
    parser.add_argument("filename")
    parser.add_argument('-e', '--epoch', help='number of epoch for sa', default='1k')
    parser.add_argument('-w', '--write', help='save solution in original file', action="store_true")
    parser.add_argument('-f', '--T0', type=float, default=2., help='start temperature for sa')
    args = parser.parse_args()
    main(args.filename, pmu.parse_with_prefix(args.epoch), args.T0, args.write)

