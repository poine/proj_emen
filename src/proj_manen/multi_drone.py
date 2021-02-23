import copy, time
import numpy as np

import proj_manen as pm
import proj_manen.utils as pm_u
import proj_manen.simulated_annealing as pm_sa

def intercept_sequences_copy(drones, sequences):
    _drones = [copy.deepcopy(_d) for _d in drones]
    _durs = [pm.intercept_sequence(_d,_s) for _d,_s in zip(_drones, sequences)]
    return _drones, _durs


def _mutate(_seqs):
    s2 = [_s.copy() for _s in _seqs]
    rng = np.random.default_rng()
    i1, i2 = rng.integers(0, len(_seqs), 2)
    j1, j2 = rng.integers(0, len(_seqs[i1])), rng.integers(0, len(_seqs[i2]))
    # FIXME, check no change?
    #pdb.set_trace()
    _tmp = s2[i1].pop(j1); s2[i2].insert(j2, _tmp)
    return s2


def format_seqs(seqs):
    return '['+']    ['.join([pm_u.format_seq(_s) for _s in seqs])+']'
def _print_sol(txt, cdur, bdur, seqs):
    print(f'{txt} {bdur:.3f} {cdur:.3f} {format_seqs(seqs)}')
    

def search(drones, targets, epochs, Tf=None, T0=2., display=0, use_native=False):
    if use_native:
        solver = pm_cpp_ext.Solver(drones, targets)# else sys.modules[__name__] # FIXME
    if Tf is None: Tf = lambda i: pm_sa._f1(T0, 0, 1e-2, 0.8*epochs, i)
    start_seqs, remaining = [[] for i in range(len(drones))], list(targets)
    rng = np.random.default_rng()
    while len(remaining)>0:
        i,j = rng.integers(0, len(remaining)), rng.integers(0, len(drones))
        start_seqs[j].append(remaining[i]); remaining.remove(start_seqs[j][-1])
    best_seqs = cur_seqs = start_seqs
    start_drones, start_durs = intercept_sequences_copy(drones, start_seqs)
    best_drones, cur_drones = start_drones, start_drones
    best_dur = cur_dur = np.max(start_durs)
    _print_sol('initial', cur_dur, best_dur, cur_seqs)
    if display>1: last_display = time.time()
    for e in range(epochs):
        T = Tf(e)
        _r = rng.random()
        _s2 = _mutate(cur_seqs)
        if use_native:
            _d2, _durs = solver.intercept_sequences_copy(drones, _s2)
        else:
            _d2, _durs = intercept_sequences_copy(drones, _s2)
        _tdur = np.max(_durs)
        acc_prob = np.exp(-(_tdur-cur_dur)/T) if _tdur > cur_dur else 0. # warning: really 1, but 0 looks nicer on plot
        if _tdur < best_dur:
            best_dur, best_drones, best_seqs = _tdur, _d2, _s2
        if  _tdur < cur_dur or _r <= acc_prob:
            cur_dur, cur_drones, cur_seqs = _tdur, _d2, _s2
        if display>1:
            now = time.time()
            if now -last_display> 1.:
                _print_sol(f'{e: 8d}', cur_dur, best_dur, cur_seqs)
                last_display = now    

        
    #print(start_durs, best_dur)
    print(f'best: {best_dur:.3f} s')
    print(f'{format_seqs(best_seqs)}')
    return best_drones, best_seqs
