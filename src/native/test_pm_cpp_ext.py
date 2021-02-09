#!/usr/bin/env python3

import time, numpy as np
import pdb

import pm_cpp_ext
import proj_manen as pm, proj_manen.utils as pmu


#
# Correct ?
#
def test1(filename = '../../data/scenario_60_6.yaml'):
    scen = pmu.Scenario(filename=filename)
    s = pm_cpp_ext.Solver()
    s.init(scen.drone, scen.targets)

    seq = [scen.targets[17]]
    #nb_t = len(scen.targets)
    #seq = np.random.permutation(scen.targets).tolist()
    _seq = [_s.name for _s in seq]
    print(_seq)
    s.run_sequence(_seq)
    c_psis = s.debug()
    #print(psis)
    # if 0:
    #     psi, dt = pm.intercept_1(scen.drone, scen.targets[0])
    #     print(f'intercept seq py {psi}, {dt}')
    #     scen.drone.add_leg(dt, psi)
    #     psi, dt = pm.intercept_1(scen.drone, scen.targets[1])
    #     print(f'intercept seq py {psi}, {dt}')
    #     scen.drone.add_leg(dt, psi)
    # else:
    pm.intercept_sequence(scen.drone, scen.targets[:nb_t])
    #print(f'dur: {scen.drone.flight_duration()}')
    #print(f'{scen.drone.psis}')
    #passed = np.allclose(psis, scen.drone.psis, rtol=1e-05, atol=1e-08)
    passed = np.allclose(c_psis, scen.drone.psis, rtol=1e-05, atol=1e-06)
    if not passed:
        print(f'{s.debug()} \n{scen.drone.psis}')
    #pdb.set_trace()
    print(f'Test passed: {passed}')

# hunting the complex root bug
def test11(filename = '../../data/scenario_120_2_2.yaml'):
    scen = pmu.Scenario(filename=filename)
    s = pm_cpp_ext.Solver(scen.drone, scen.targets)
    for i in range(1):
        seq = np.random.permutation(scen.targets).tolist()#[:nb_tg]
        _seq = [_s.name-1 for _s in seq]
        c_dur = s.run_sequence(_seq);c_psis = s.debug()
        print(f'intercepted {len(c_psis)} targets')

# compare python and c cost function on a set of random permutations
def test2(filename = '../../data/scenario_60_6.yaml', nb_tg=60, ntest=100):
    scen = pmu.Scenario(filename=filename)
    s = pm_cpp_ext.Solver()
    s.init(scen.drone, scen.targets)

    for i in range(ntest):
        seq = np.random.permutation(scen.targets).tolist()[:nb_tg]
        _seq = [_s.name-1 for _s in seq]
        #print(_seq)
        c_dur = s.run_sequence(_seq);c_psis = s.debug()
        py_drone, py_dur = pm.intercept_sequence_copy(scen.drone, seq)

        #pdb.set_trace()
        c_psis1 = [pmu.norm_angles_mpi_pi(_psi) for _psi in c_psis]
        p_psis1 = [pmu.norm_angles_mpi_pi(_psi) for _psi in py_drone.psis]
        passed1 = np.allclose(c_psis1, p_psis1, rtol=1e-02, atol=1e-03) # rtol=1e-05, atol=1e-08
        #passed1 = np.allclose(c_psis, py_drone.psis, rtol=1e-05, atol=1e-06) # rtol=1e-05, atol=1e-08
        passed2 = np.allclose(c_dur, py_dur, rtol=1e-05, atol=1e-06)
        print(f'Test passed: {passed1} {passed2}')
        if not passed1:
            #print(f'{c_psis} \n{py_drone.psis}')
            for i in range(nb_tg):
                if abs(c_psis[i]-py_drone.psis[i]) > 1e-5: print(f'failed idx {i} {c_psis1[i]} {p_psis1[i]}')
            print([_s.name for _s in seq])
        if not passed2:
            print(f'durations (c/py): {c_dur}, {py_dur}')
        #pdb.set_trace()

#
# intercept a sequence from a scenario
#
def test22(filename = '../../data/scenario_60_6.yaml'):
    
    #seq = [30, 47, 12, 20, 39, 36, 28, 29, 18, 2, 41, 24, 35, 54, 58, 23, 53, 11, 4, 15, 10, 9, 34, 43, 44, 45, 42, 14, 37, 17, 19, 52, 8, 33, 48, 27, 56, 5, 55, 21, 59, 26, 40, 50, 49, 7, 22, 31, 38, 25, 57, 32, 3, 60, 6, 13, 51, 46, 16, 1]
    seq = [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17]
    
    scen = pmu.Scenario(filename=filename)
    s = pm_cpp_ext.Solver()
    s.init(scen.drone, scen.targets)
    py_seq = [scen.targets[_n-1] for _n in seq]
    py_drone, py_dur = pm.intercept_sequence_copy(scen.drone, py_seq)
    print(f'{py_dur}')
    c_drone, c_dur = s.intercept_sequence_copy(scen.drone, py_seq)
    print(f'{c_dur}')
    pdb.set_trace()

    
#
# Profiling: x200?
#
def test3(filename = '../../data/scenario_240_6.yaml', ntest=1000):
    scen = pmu.Scenario(filename=filename)
    s = pm_cpp_ext.Solver(scen.drone, scen.targets)
    seq = [_s.name for _s in scen.targets]

    _start1 = time.perf_counter()
    for i in range(ntest):
        s.run_sequence(seq)
    _end1 = time.perf_counter()
    _dt1 = _end1-_start1
    print(f'{ntest} evaluations in C took {_dt1:.3f} s')

    _start2 = time.perf_counter()
    for i in range(ntest):
        pm.intercept_sequence(scen.drone, scen.targets)
        scen.drone.clear_traj()
    _end2 = time.perf_counter()
    _dt2 = _end2-_start2 
    print(f'{ntest} evaluations in Python took {_dt2:.3f} s')
    print(f'improvement {_dt2/_dt1:.1f}')
    
#
# search exhaustive
#
def test4(filename = '../../data/scenario_9_6.yaml'):
    scen = pmu.Scenario(filename=filename)
    s = pm_cpp_ext.Solver()
    s.init(scen.drone, scen.targets)
    n_targ = len(scen.targets); n_seq= np.math.factorial(n_targ)
    print(f'searching in all {n_targ} targets permutations ({n_seq:.2e})')
    _start = time.perf_counter()
    best_cost, best_seq = s.run_exhaustive()
    _end = time.perf_counter()
    dt = _end-_start; ips = n_seq/dt
    print(f'best cost: {best_cost}')
    print(best_seq)
    print(f'search in C took {dt:.3f}s ({ips:.2e} cost_evals/s)')

    
#test1()
#test11()  # hunting complex root bug: float overflow
#test2()
#test22()
#test3()   # profiling cost evaluation
test4()   # exhaustive search
