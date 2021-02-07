import copy

'''
Some glue for the cython c++ interface
'''

try:
    import pm_cpp_ext
except ImportError:
    print('proj_manen.simulated_annealing: failed to import native code')


def search_exhaustive(drone, targets):
    s = pm_cpp_ext.Solver()
    s.init(drone, targets)
    best_dur, _best_seq = s.run_all()
    drone = copy.deepcopy(drone)
    drone.ts.append(best_dur)
    best_seq = [targets[_i] for _i in _best_seq]
    return drone, best_seq

