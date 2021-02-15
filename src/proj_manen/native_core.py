import copy, numpy as np

'''
Some glue for the cython c++ interface
'''

try:
    import pm_cpp_ext
except ImportError:
    print('')
    print('## WARNING: proj_manen.native_core: failed to import native library.')
    print('    you will not be able to use the C++ backend which is needed for increased performances.')
    print('    See the documentation for how to build and use it.')
    print('')


def search_exhaustive(drone, targets):
    s = pm_cpp_ext.Solver(drone, targets)
    best_dur, _best_seq = s.search_exhaustive()
    drone = copy.deepcopy(drone)
    drone.ts.append(best_dur)  # Warning: we only update flight time
    best_seq = [targets[_i] for _i in _best_seq]
    return drone, best_seq

def search_sa(drone, targets, start_seq, nepoch, T0=1., display=1):
    s = pm_cpp_ext.Solver(drone, targets)
    if start_seq is None: start_seq = np.random.permutation(targets).tolist()
    _start_seq = [_t.name-1 for _t in start_seq]
    best_dur, _best_seq = s.search_sa(_start_seq, nepoch, T0, display)
    drone = copy.deepcopy(drone)
    drone.ts.append(best_dur)  # Warning: we only update flight time
    best_seq = [targets[_i] for _i in _best_seq]
    return drone, best_seq
    
