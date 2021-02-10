import copy

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
    best_dur, _best_seq = s.run_exhaustive()
    drone = copy.deepcopy(drone)
    drone.ts.append(best_dur)  # Warning: we only update flight time
    best_seq = [targets[_i] for _i in _best_seq]
    return drone, best_seq

