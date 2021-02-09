#
# simulated annealing
#

import numpy as np, itertools, time
import pdb

import proj_manen as pm, proj_manen.utils as pmu

def _names_of_seq(_s): return [_t.name.split('_')[-1] for _t in _s]
def _names_of_seq2(_s): foo = [int(_t.name.split('_')[-1]) for _t in _s]; return '-'.join([f'{_f:02d}' for _f in foo])
def _print_sol(_e, _t, _bd, _cd, _s):
    if len(_s) < 61:
        print(f'{_e: 8d}  {_t: 8.2f}  {_bd: 8.2f}  {_cd: 8.2f}   {pmu.format_seq(_s)}')
    else:
        print(f'{_e: 8d}  {_t: 8.2f}  {_bd: 8.2f}  {_cd: 8.2f}')

def _mutate(_seq):  # swaping two random stages
    _seq2 = _seq.copy()
    i1 = np.random.randint(0, high=len(_seq))
    i2 = np.random.randint(0, high=len(_seq)-1)
    while i2==i1: i2 = np.random.randint(0, high=len(_seq)-1)
    _foo = _seq2.pop(i1); _seq2.insert(i2, _foo)
    return _seq2

def _mutate2(_seq):  # swaping two adjacent stages
    _seq2 = _seq.copy()
    i1 = np.random.randint(0, high=len(_seq))
    i2 = i1-1 if i1>0 else i1+1
    _foo = _seq2.pop(i1); _seq2.insert(i2, _foo)
    return _seq2

def _aff(a0, a1, n, i): return a0 + (a1-a0)*i/n
def _exp(e0, n, i): return e0*np.exp(-i/n)
def _f1(a0, i0, a1, i1, i): return a0 if i<=i0 else a0+(a1-a0)*(i-i0)/(i1-i0) if i <= i1 else a1


try:
    import pm_cpp_ext
except ImportError:
    print('proj_manen.simulated_annealing: failed to import native code')


def search(drone, targets, start_dur=None, start_seq=None, ntest=1000, debug=False, Tf=None, display=0, T0=2., use_native=False):
    if use_native:
        solver = pm_cpp_ext.Solver(drone, targets)
    else:
        solver = pm
    if display>0:
        print(f'running simulated annealing with {len(targets)} targets for {ntest:.1e} epochs')
        print(f'  ({ntest/np.math.factorial(len(targets)):.2e} search space coverage)')
        if Tf is not None: print('   custom temperature control')
        else: print(f'  default linear temperature (T0={T0})')
    start_seq = np.random.permutation(targets).tolist()
    start_drone, start_dur = pm.intercept_sequence_copy(drone, start_seq)
    
    best_drone, cur_drone = start_drone, start_drone
    best_seq = cur_seq = start_seq
    best_dur = cur_dur = start_dur
    if display>1: last_display = time.perf_counter()
    if Tf is None: Tf = lambda i: _f1(T0, ntest/10, 1e-2, 8*ntest/10, i)
    if debug: all_durs, kept_durs, Paccept = np.zeros(ntest) , np.zeros(ntest), np.zeros(ntest) 
    for i in range(ntest):
        _s2 = _mutate(cur_seq)
        _d2, _dur = solver.intercept_sequence_copy(drone, _s2)
        if debug: all_durs[i] = _dur
        if _dur < best_dur:
            best_dur, best_drone, best_seq = _dur, _d2, _s2
        T = Tf(i)
        acc_prob = np.exp(-(_dur-cur_dur)/T) if _dur > cur_dur else 0. # warning 1, but 0 looks nicer on plot
        if debug: Paccept[i] = acc_prob
        _r = np.random.uniform(low=0, high=1.)
        if _dur < cur_dur or _r <= acc_prob:
            cur_dur, cur_drone, cur_seq = _dur, _d2, _s2
            if display>1 and time.perf_counter()-last_display> 1.:
                _print_sol(i, T, best_dur, cur_dur, cur_seq)
                last_display = time.perf_counter()
        if debug: kept_durs[i] = cur_dur

    #__best_drone, __best_dur =  pm.intercept_sequence_copy(drone, best_seq)
    #__best_drone2, __best_dur2 =  s.intercept_sequence_copy(drone, best_seq)
    if display>0:
        print(f'  best solution')
        #pdb.set_trace()
        _print_sol(i, T, best_dur, cur_dur, cur_seq)
    if debug: return best_drone, best_seq, all_durs, kept_durs, Paccept
    else: return best_drone, best_seq

