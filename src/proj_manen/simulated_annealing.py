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

# mutation operator
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

# functions for Temperature
def _aff(a0, a1, n, i): return a0 + (a1-a0)*i/n
def _step(a0, a1, n, i): return a0 if i<=n else a1
def _2aff(a0, a1, n1, a2, a3, n2, i): return a0 + (a1-a0)*i/n1 if i <= n1 else a2 + (a3-a2)*(i-n1)/(n2-n1)
def _exp(e0, n, i): return e0*np.exp(-i/n)
def _f1(a0, i0, a1, i1, i): return a0 if i<=i0 else a0+(a1-a0)*(i-i0)/(i1-i0) if i <= i1 else a1

try:
    import pm_cpp_ext
except ImportError:
    print('proj_manen.simulated_annealing: failed to import native code')


def search(drone, targets, start_seq=None, epochs=1000, debug=False, Tf=None, display=0, T0=2., use_native=False):
    solver = pm_cpp_ext.Solver(drone, targets) if use_native else pm
    if display>0:
        print(f'running simulated annealing with {len(targets)} targets for {epochs:.1e} epochs')
        print(f'  ({epochs/np.math.factorial(len(targets)):.2e} search space coverage)')
        print('  custom temperature control'  if Tf is not None else f'  default linear temperature (T0={T0})')
    if start_seq is None: start_seq = np.random.permutation(targets).tolist()
    start_drone, start_dur = pm.intercept_sequence_copy(drone, start_seq)
    best_drone, cur_drone = start_drone, start_drone
    best_seq = cur_seq = start_seq
    best_dur = cur_dur = start_dur
    if display>0:
        print(f'  start solution')
        _print_sol(0, T0, best_dur, cur_dur, cur_seq)
    if display>1: last_display = time.perf_counter()
    if Tf is None: Tf = lambda i: _f1(T0, 0, 1e-2, 0.8*epochs, i)
    if debug: all_durs, cur_durs, Paccept, max_durs = (np.zeros(epochs) for i in range(4))
    for i in range(epochs):
        T = Tf(i)
        _r = np.random.uniform(low=0, high=1.)
        max_dur = cur_dur + T*np.log(_r)
        _s2 = _mutate(cur_seq)
        # BROKEN!! _d2, _dur = solver.intercept_sequence_copy_threshold(drone, _s2, max_dur)
        _d2, _dur = solver.intercept_sequence_copy(drone, _s2)
        acc_prob = np.exp(-(_dur-cur_dur)/T) if _dur > cur_dur else 0. # warning 1, but 0 looks nicer on plot
        if debug:
            max_durs[i] = max_dur; all_durs[i] = _dur; Paccept[i] = acc_prob
        if _dur < best_dur:
            best_dur, best_drone, best_seq = _dur, _d2, _s2
        if _dur < cur_dur or _r <= acc_prob:
            cur_dur, cur_drone, cur_seq = _dur, _d2, _s2
            if display>1 and time.perf_counter()-last_display> 1.:
                _print_sol(i, T, best_dur, cur_dur, cur_seq)
                last_display = time.perf_counter()
        if debug: cur_durs[i] = cur_dur

    if display>0:
        print(f'  best solution')
        _print_sol(i, T, best_dur, cur_dur, cur_seq)
    if debug: return best_drone, best_seq, all_durs, cur_durs, Paccept, max_durs
    else: return best_drone, best_seq

