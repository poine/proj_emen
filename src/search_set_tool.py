#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Manipulate a search set
'''

import sys, argparse, os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.simulated_annealing as pm_sa

def create_search_set(scen_filename, nb_searches, epochs, cache_filename):
    scen = pmu.Scenario(filename=scen_filename)
    cost_by_ep, seq_by_ep = [],[]
    for _ep in epochs:
        print(f'-{_ep} epochs')
        _drones, _seqs = [],[]
        for i in range(nb_searches):
            _drone, _seq = pm_sa.search(scen.drone, scen.targets, ntest=_ep, display=1)
            _drones.append(_drone); _seqs.append(_seq)
            print(f' run {i: 3d}: {_drone.flight_duration():.2f}s')
        cost_by_run = [_d.flight_duration() for _d in _drones]
        cost_by_ep.append(cost_by_run)
        seq_by_ep.append([pmu.format_seq(_s) for _s in _seqs])
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
        plt.hist(c, label=f'{e}', alpha=0.6)
    pmu.decorate(plt.gca(), xlab='time in s', legend=True)

    
def main(set_filename, show, create):
    #pdb.set_trace()
    if show:
        data = load_search_set(cache_filename=set_filename)
        plot_search_set(*data, set_filename)
        plt.show()
    if create:
        scen_filename = pmu.ressource('data/scenario_60_6.yaml')
        nb_searches, epochs = 50, [int(5e3), int(1e4), int(2e4), int(5e4), int(1e5)]
        print(f'about to run {nb_searches} searches for {epochs} epochs on {scen_filename}')
        print(f' results will be stored in {set_filename}')
        data = create_search_set(scen_filename, nb_searches, epochs, set_filename)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manipulate s search set.')
    parser.add_argument('set_filename')
    parser.add_argument('-c', '--create', help='create', action='store_true')
    parser.add_argument('-x', '--show', help='show', action='store_true')
    args = parser.parse_args()
    main(args.set_filename, args.show, args.create)
