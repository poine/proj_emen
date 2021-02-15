#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Unit test for multiprocessing
'''
import argparse, os, time, datetime, numpy as np, matplotlib.pyplot as plt
import multiprocessing as mp
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.animations as pma, proj_manen.simulated_annealing as pm_sa
import proj_manen.native_core as pm_nc


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

#def run_sa(idx, scen, start_seq, epochs):#, queue):
def run_sa(args):#, queue):
    idx, scen, start_seq, epochs, queue = args
    #info('run_sa')
    #print(f'{idx} starting')# on: {scen.name} {epochs:.1e}')
    #res = pm_sa.search(scen.drone, scen.targets, start_seq, epochs=epochs, display=0, use_native=True)
    res = pm_nc.search_sa(scen.drone, scen.targets, start_seq, epochs, display=0)
    print(f'{idx:03d} -> {res[0].flight_duration():.3f} s')
    #queue.task_done()
    queue.put((res[0].flight_duration(), res[1]))

def main(scen_filename, epochs, nruns, parallel, save=False):
    print(f'Running {nruns} runs of {epochs:.1e} epochs on {scen_filename}')
    print(f'Number of cpu: {mp.cpu_count()} Number of parallel tasks {parallel}')
    #info('main line')
    _start = time.time()
    scen = pmu.Scenario(filename=scen_filename)
    start_seqs = [np.random.permutation(scen.targets).tolist() for _i in range(nruns)]
    pool = mp.Pool(parallel)
    m = mp.Manager()
    queue = m.Queue()
    _d = [(_i, scen, _s, epochs, queue) for _i, _s in enumerate(start_seqs)]
    pool.map(run_sa, _d)
    pool.close()
    pool.join()
    _end = time.time()
    print(f'computed search set in {datetime.timedelta(seconds=_end-_start)} (h::m::s)')
    res = [queue.get() for i in range(nruns)]
    _durs, _seqs = [_r[0] for _r in res], [_r[1] for _r in res]
    #with np.printoptions(precision=2, suppress=True):
    #    print(f'{_durs}')
    print(f'{pmu.fmt_mmmm(pmu.mmmm(_durs))}')
    best_idx = np.argmin(_durs)
    best_dur, best_seqn = _durs[best_idx], _seqs[best_idx]
    print(f'best : {best_dur:.4f} s')
    if save:
        scen.add_solution('best', best_dur, best_seqn)
        scen.save(scen_filename) 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simulated annealing in parallel.')
    parser.add_argument('scen_filename')
    parser.add_argument('-e', '--epochs', help='epochs', default='1k')
    parser.add_argument('-t', '--nb_run', help='nb runs', type=int, default=8)
    parser.add_argument('-J', '--parallel', help='nb runs', type=int, default=mp.cpu_count())
    parser.add_argument('-w', '--overwrite', help='overwrite', action='store_true')
    #parser.add_argument('-s', '--scen_filename', help='scenario filename')
    args = parser.parse_args()
    epochs = pmu.parse_with_prefix(args.epochs)
    main(args.scen_filename, epochs, args.nb_run, args.parallel, args.overwrite)
    
