#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Runs a number of searches on a given scenario, cache and display results
'''

import argparse, os, time, datetime, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.simulated_annealing as pm_sa
import proj_manen.native_core as pm_nc


def _display_run(_ep, id_run, nb_run, _costs):
    _min, _med, _max = np.min(_costs), np.median(_costs), np.max(_costs)
    print(f' {id_run+1:03d}/{nb_run} {_costs[-1]: 8.3f}s min/med/max {_min:.3f} / {_med:.3f} / {_max:.3f}')

def _display_ep(_ep, _costs, cpu_elapsed):
    _min, _med = np.min(_costs), np.median(_costs)
    print(f'{_ep} epochs, min {_min:.3f}s, cpu {datetime.timedelta(seconds=cpu_elapsed)} (h:m:s)')
    
def create_search_set(scen_filename, nb_searches, epochs, cache_filename, T0=2.):
    time_tags = []
    time_tags.append(time.perf_counter())
    scen = pmu.Scenario(filename=scen_filename)
    cost_by_ep, seq_by_ep = [],[]
    for _ep in epochs:
        print(f'-{_ep:e} epochs')
        _drones, _seqs, _costs = [],[],[]
        for i in range(nb_searches):
            #_drone, _seq = pm_sa.search(scen.drone, scen.targets, epochs=_ep, display=0)
            _drone, _seq = pm_sa.search(scen.drone, scen.targets, epochs=_ep, display=0, T0=T0, use_native=True)
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
    return np.asarray(cost_by_ep), np.as_array(seq_by_ep), np.asarray(epochs)
        
def load_search_set(cache_filename):
    data = np.load(cache_filename)
    cost_by_ep, seq_by_ep, epochs = data['cost_by_ep'], data['seq_by_ep'], data['epochs'] 
    return cost_by_ep, seq_by_ep, epochs

def plot_search_set(cost_by_ep, seq_by_ep, epochs, window_title, skip=0):
    for e, c in zip(epochs[skip:], cost_by_ep[skip:]):
        plt.hist(c, label=f'{e/1000} k epochs', alpha=0.6)
    pmu.decorate(plt.gca(), title=window_title, xlab='time in s', legend=True)
    plt.gcf().canvas.set_window_title(window_title)

def analyze_search_set(cost_by_ep, seq_by_ep, epochs, ds_filename, add_best=False, add_good=False, scen_filename=None, overwrite=False):
    best_idx = np.argmin(cost_by_ep)
    best_dur, best_seqn = cost_by_ep.flatten()[best_idx], seq_by_ep.flatten()[best_idx]
    print(f'min cost {best_dur:.3f} {best_seqn}')
    good_range = 1.01
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
    if add_good:
        for _i, (_d, _seqn) in enumerate(zip(good_costs, good_seqns)):
            scen.add_solution(f'best__{_i}', _d, tg_seq_of_names(_seqn))
    if overwrite:
        scen.save(scen_filename)


def merge_search_set(d1, d2):
    (_cbe1, _sbe1, _e1), (_cbe2, _sbe2, _e2) = d1, d2
    epochs = np.append(_e1, _e2)
    cost_by_ep = np.append(_cbe1, _cbe2, axis=0)
    seq_by_ep = np.append(_sbe1, _sbe2, axis=0)
    #pdb.set_trace()
    return cost_by_ep, seq_by_ep, epochs


def main(set_filenames, create, analyze, show, scen_filename, nb_searches, epochs, overwrite, add_best, add_good):
    if create:
        print(f'## About to run {nb_searches} searches for {epochs} epochs on {scen_filename}')
        print(f'## results will be stored in {set_filenames[0]}')
        data = create_search_set(scen_filename, nb_searches, epochs, set_filenames[0])
    else:
        data = load_search_set(cache_filename=set_filenames[0])
        for sf in set_filenames[1:]:
            data = merge_search_set(data, load_search_set(cache_filename=sf))

    if analyze:
        analyze_search_set(*data, set_filenames[0], add_best=add_best, add_good=add_good, scen_filename=scen_filename, overwrite=overwrite)
        
    if show:
        plot_search_set(*data, os.path.basename(set_filenames[0]), skip=0)
        plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manipulate s search set.')
    parser.add_argument('set_filename', nargs='*')
    parser.add_argument('-x', '--show', help='show', action='store_true')
    parser.add_argument('-c', '--create', help='create', action='store_true')
    parser.add_argument('-s', '--scen_filename', help='scenario filename')
    parser.add_argument('-t', '--nb_run', help='nb runs', type=int)
    parser.add_argument('-e', '--epochs', help='epochs')
    parser.add_argument('-a', '--analyze', help='analyze', action='store_true')
    parser.add_argument('-b', '--add_best', help='edit', action='store_true')
    parser.add_argument('-g', '--add_good', help='edit', action='store_true')
    parser.add_argument('-w', '--overwrite', help='overwrite', action='store_true')

    args = parser.parse_args()
    epochs = [int(float(_)) for _ in args.epochs.split(',')] if args.epochs is not None else [] # example: 5e3,1e4
    #pdb.set_trace()
    #epochs = [int(1e3), int(5e3)]#, int(1e4), int(2e4), int(5e4), int(1e5)]
    main(args.set_filename, args.create, args.analyze, args.show, args.scen_filename, args.nb_run, epochs, args.overwrite, args.add_best, args.add_good)
