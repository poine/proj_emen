#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Unit test for multiprocessing
'''
import argparse, os, time, datetime, numpy as np
import multiprocessing as mp
import pdb

import proj_manen.utils as pm_u
import proj_manen.scenarios as pm_sc
import proj_manen.simulated_annealing as pm_sa
import proj_manen.native_core as pm_nc
import proj_manen.multi_drone as pm_md

# Main function for sub processes
def run_sa(args):
    idx, scen, start_seq, epochs, T0, queue, method = args
    #print(f'{idx} starting')# on: {scen.name} {epochs:.1e}')
    if   method=='sa2': ## Python sa with C++ cost evaluation
        res = pm_sa.search(scen.drone, scen.targets, start_seq, epochs=epochs, display=0, use_native=True)
    elif method=='sa3': ## C++ sa (with C++ cost evaluation)
        res = pm_nc.search_sa(scen.drone, scen.targets, start_seq, epochs, display=0, T0=T0)
    elif method == 'mdsa2': ## multi drones, Python sa with C++ cost evaluation
        res = pm_md.search(scen.drones, scen.targets, epochs, T0=T0, display=0, use_native=True)
    if type(res[0])!=list: res[0] = [res[0]] # FIXME: make that homogeneous
    fd = np.max([_d.flight_duration() for _d in res[0]])
    print(f'{idx:03d} -> {fd:.3f} s')
    queue.put((fd, res[1]))

# Runs nruns simulated annealing searches over epochs epochs. Searches are handled by a pool of subprocesses,
# parallel of them running concurrently.
def main(scen_filename, epochs, nruns, parallel, T0=2., method='sa3', save=False, save_name='best'):
    print(f'Running {nruns} runs of {epochs:.1e} epochs on {scen_filename}')
    print(f'Number of cpu: {mp.cpu_count()} Number of parallel tasks {parallel}')
    _start = time.time()
    scen = pm_sc.Scenario(filename=scen_filename)
    start_seqs = [np.random.permutation(scen.targets).tolist() for _i in range(nruns)]
    pool = mp.Pool(parallel)
    mngr = mp.Manager()
    queue = mngr.Queue()
    sp_args = [(_i, scen, _ss, epochs, T0, queue, method) for _i, _ss in enumerate(start_seqs)]
    pool.map(run_sa, sp_args)
    pool.close()
    pool.join()
    _end = time.time()
    print(f'computed search set in {datetime.timedelta(seconds=_end-_start)} (h::m::s)')
    res = [queue.get() for i in range(nruns)]
    _durs, _seqs = [_r[0] for _r in res], [_r[1] for _r in res]
    print(f'{pm_u.fmt_mmmm(pm_u.mmmm(_durs))}')
    best_idx = np.argmin(_durs)
    best_dur, best_seqn = _durs[best_idx], _seqs[best_idx]
    print(f'best : {best_dur:.4f} s')
    if save:
        scen.add_solution(save_name, best_dur, best_seqn)
        scen.save(scen_filename) 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simulated annealing in parallel.')
    parser.add_argument('scen_filename')
    parser.add_argument('-e', '--epochs', help='epochs', default='1k')
    parser.add_argument('-t', '--nb_run', help='nb runs', type=int, default=8)
    parser.add_argument('-m', '--method', default='sa3', help='search method: sa, sa2, sa3, mdsa2')
    parser.add_argument('-f', '--T0', type=float, default=2., help='start temperature for sa')
    parser.add_argument('-J', '--parallel', help='nb runs', type=int, default=mp.cpu_count())
    parser.add_argument('-w', '--write', help='save best solution to scenario', action='store_true')
    parser.add_argument('-n', '--sol_name', help='name of solution to save as', default='best')
    args = parser.parse_args()
    epochs = pm_u.parse_with_prefix(args.epochs)
    main(args.scen_filename, epochs, args.nb_run, args.parallel, args.T0, args.method, args.write, args.sol_name)
    
