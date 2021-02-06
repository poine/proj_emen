#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Manipulate a search set
'''

import argparse, os, time, datetime, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.simulated_annealing as pm_sa


def _display_run(_ep, id_run, nb_run, _costs):
    foo = [f'{_:.3f}' for _ in _costs]
    _min, _med, _max = np.min(_costs), np.median(_costs), np.max(_costs)
    print(f'{_ep} {id_run+1:03d}/{nb_run} min/med/max {_min:.3f} / {_med:.3f} / {_max:.3f}')

def _display_ep(_ep, _costs, cpu_elapsed):
    _min, _med = np.min(_costs), np.median(_costs)
    print(f'{_ep} epochs, min {_min:.3f}s, cpu {datetime.timedelta(seconds=cpu_elapsed)} (h:m:s)')
    
def create_search_set(scen_filename, nb_searches, epochs, cache_filename):
    time_tags = []
    time_tags.append(time.perf_counter())
    scen = pmu.Scenario(filename=scen_filename)
    cost_by_ep, seq_by_ep = [],[]
    for _ep in epochs:
        print(f'-{_ep} epochs')
        _drones, _seqs, _costs = [],[],[]
        for i in range(nb_searches):
            _drone, _seq = pm_sa.search(scen.drone, scen.targets, ntest=_ep, display=0)
            _drones.append(_drone); _seqs.append(_seq)
            _costs.append(_drone.flight_duration())
            _display_run(_ep, i, nb_searches, _costs)
        cost_by_ep.append(_costs)
        seq_by_ep.append([pmu.format_seq(_s) for _s in _seqs])
        time_tags.append(time.perf_counter())
        _display_ep(_ep, _costs, time_tags[-1]-time_tags[-2])
    print(f'computed search set in {datetime.timedelta(seconds=time_tags[-1]-time_tags[0])}')
    print(f'saving to {cache_filename}')
    np.savez(cache_filename, cost_by_ep=cost_by_ep, seq_by_ep=seq_by_ep, epochs=epochs)
    return cost_by_ep, seq_by_ep, epochs
        
def load_search_set(cache_filename):
    data = np.load(cache_filename+'.npz')
    cost_by_ep, seq_by_ep, epochs = data['cost_by_ep'], data['seq_by_ep'], data['epochs'] 
    return cost_by_ep, seq_by_ep, epochs

def update_search_set(cache_filename1, cache_filename2, out_filename=None):
    _cbe1, _sbe1, _e1 = create_or_load_search_set(None, None, None, cache_filename1, False)
    _cbe2, _sbe2, _e2 = create_or_load_search_set(None, None, None, cache_filename2, False)
    epochs = np.append(_e1, _e2)
    cost_by_ep = np.append(_cbe1, _cbe2, axis=0)
    seq_by_ep = np.append(_sbe1, _sbe2, axis=0)
    if out_filename is not None:
        np.savez(out_filename, cost_by_ep=cost_by_ep, seq_by_ep=seq_by_ep, epochs=epochs)
    return cost_by_ep, seq_by_ep, epochs

def plot_search_set(cost_by_ep, seq_by_ep, epochs, window_title):
    for e, c in zip(epochs, cost_by_ep):
        plt.hist(c, label=f'{e/1000} k epochs', alpha=0.6)
    pmu.decorate(plt.gca(), xlab='time in s', legend=True)
    plt.gcf().canvas.set_window_title(window_title)

def analyze_search_set(cost_by_ep, seq_by_ep, epochs, ds_filename, add_best=False, scen_filename=None, overwrite=False):
    best_idx = np.argmin(cost_by_ep)
    best_dur, best_seqn = cost_by_ep.flatten()[best_idx], seq_by_ep.flatten()[best_idx]
    print(f'min cost {best_dur:.3f} {best_seqn}')
    good_range = 1.05
    good_idx = cost_by_ep <  best_dur*good_range
    good_costs, good_seqns = cost_by_ep[good_idx], seq_by_ep[good_idx]
    print(f'found {len(good_costs)} solutions within {(good_range-1)*100:.2f}% of optimal')
    print(f'{good_costs}')
    if scen_filename is not None:
        scen = pmu.Scenario(filename=scen_filename)
        target_by_name = {str(_t.name):_t for _t in scen.targets}
    def tg_seq_of_names(names):
        target_names = [f'{_:d}' for _ in [int(_) for _ in names.split('-')]]
        return [target_by_name[_tn] for _tn in target_names]
        
    if add_best:
        scen.add_solution('best', best_dur, tg_seq_of_names(best_seqn))
    #if add_good:
    #    for _i, (_d, _seqn) in enumerate(zip(good_costs, good_seqns)):
    #        scen.add_solution(f'best{_i}', _d, tg_seq_of_names(_seqn))
    if overwrite:
        scen.save(scen_filename)
    
def main(set_filename, create, analyze, show, scen_filename, nb_searches, epochs, overwrite, add_best):
    #pdb.set_trace()
    if create:
        print(f'## About to run {nb_searches} searches for {epochs} epochs on {scen_filename}')
        print(f'## results will be stored in {set_filename}')
        data = create_search_set(scen_filename, nb_searches, epochs, set_filename)
    else:
        data = load_search_set(cache_filename=set_filename)
    if analyze:
        analyze_search_set(*data, set_filename, add_best=add_best, scen_filename=scen_filename, overwrite=overwrite)
        
    if show:
        plot_search_set(*data, set_filename)
        plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manipulate s search set.')
    parser.add_argument('set_filename')
    parser.add_argument('-x', '--show', help='show', action='store_true')
    parser.add_argument('-c', '--create', help='create', action='store_true')
    parser.add_argument('-s', '--scen_filename', help='scenario filename')
    parser.add_argument('-t', '--nb_run', help='nb runs', type=int)
    parser.add_argument('-e', '--epochs', help='epochs')
    parser.add_argument('-a', '--analyze', help='analyze', action='store_true')
    parser.add_argument('-b', '--add_best', help='edit', action='store_true')
    parser.add_argument('-w', '--overwrite', help='overwrite', action='store_true')

    args = parser.parse_args()
    epochs = [int(float(_)) for _ in args.epochs.split(',')] if args.epochs is not None else [] # example: 5e3,1e4
    #pdb.set_trace()
    #epochs = [int(1e3), int(5e3)]#, int(1e4), int(2e4), int(5e4), int(1e5)]
    main(args.set_filename, args.create, args.analyze, args.show, args.scen_filename, args.nb_run, epochs, args.overwrite, args.add_best)
