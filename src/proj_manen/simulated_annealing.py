#
# simulated annealing
#

import numpy as np, itertools
import pdb

import proj_manen as pm

def _names_of_seq(_s): return [_t.name.split('_')[-1] for _t in _s]
def _names_of_seq2(_s): foo = [int(_t.name.split('_')[-1]) for _t in _s]; return '-'.join([f'{_f:02d}' for _f in foo])
def _print_sol(_bd, _cd, _s): print(f'{_bd: 8.2f} {_cd: 8.2f}: {_names_of_seq2(_s)}')

def _mutate(_seq):  # swaping to random stages
    _seq2 = _seq.copy()
    i1 = np.random.randint(0, high=len(_seq))
    i2 = np.random.randint(0, high=len(_seq)-1)
    while i2==i1: i2 = np.random.randint(0, high=len(_seq)-1)
    _foo = _seq2.pop(i1); _seq2.insert(i2, _foo)
    return _seq2

def _mutate2(_seq):  # swaping to adjacent stages
    _seq2 = _seq.copy()
    i1 = np.random.randint(0, high=len(_seq))
    i2 = i1-1 if i1>0 else i1+1
    _foo = _seq2.pop(i1); _seq2.insert(i2, _foo)
    return _seq2


def search(drone, targets, start_dur=None, start_seq=None, ntest=1000, debug=False, Tf=None):
    if debug: print(f'running simulated annealing for {ntest} epoch')
    start_seq = np.random.permutation(targets).tolist()
    start_drone, start_dur = pm.intercept_sequence_copy(drone, start_seq)
    
    best_drone, cur_drone = start_drone, start_drone
    best_seq = cur_seq = start_seq
    best_dur = cur_dur = start_dur
    if Tf is None: Tf = lambda i: ntest/(i+1)
    if debug: all_durs, kept_durs, Paccept = np.zeros(ntest) , np.zeros(ntest), np.zeros(ntest) 
    for i in range(ntest):
        _s2 = _mutate(cur_seq)
        _d2, _dur = pm.intercept_sequence_copy(drone, _s2)
        if debug: all_durs[i] = _dur
        if _dur < best_dur:
            best_dur, best_drone, best_seq = _dur, _d2, _s2
            #_print_sol(best_dur, best_seq)
        T = Tf(i)
        acc_prob = np.exp(-(_dur-cur_dur)/T) if _dur > cur_dur else 0. # warning 1
        if debug: Paccept[i] = acc_prob
        _r = np.random.uniform(low=0, high=1.)
        if _dur < cur_dur or _r <= acc_prob:
            cur_dur, cur_drone, cur_seq = _dur, _d2, _s2
            #_print_sol(cur_dur, cur_seq)
        if debug: kept_durs[i] = cur_dur

    if debug: return best_drone, best_seq, all_durs, kept_durs, Paccept
    else: return best_drone, best_seq
