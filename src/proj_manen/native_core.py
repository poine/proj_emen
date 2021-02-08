import copy

'''
Some glue for the cython c++ interface
'''

try:
    import pm_cpp_ext
except ImportError:
    print('proj_manen.native_core: failed to import native library')


def search_exhaustive(drone, targets):
    s = pm_cpp_ext.Solver(drone, targets)
    best_dur, _best_seq = s.run_exhaustive()
    drone = copy.deepcopy(drone)
    drone.ts.append(best_dur)  # Warning: we only update flight time
    best_seq = [targets[_i] for _i in _best_seq]
    return drone, best_seq

