# distutils: language = c++
#
#
#  cython interface to the C++ target interception  code 
#
# Author: Poine-2021
#

#cython: language_level=3

import numpy as np, copy

from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "pm/pm.h":
    cdef cppclass c_Solver "Solver":
        c_Solver()
        bool init(float* dp, float dv, vector[float] tx, vector[float] ty, vector[float] tv, vector[float] th)
        float run(vector[int] seq)
        float run_exhaustive(vector[int] &best_seq)
        vector[float] get_psis()

cdef class Solver:
    cdef c_Solver *thisptr

    def __cinit__(self):
        self.thisptr = new c_Solver()

    def init(self, drone, targets):
        cdef float dp[2]
        cdef float dv
        dp[0] = drone.p0[0]
        dp[1] = drone.p0[1]
        dv = drone.v
        cdef vector[float] tx
        cdef vector[float] ty
        cdef vector[float] tv
        cdef vector[float] th
        for _t in targets:
            tx.push_back(_t.x0)
            ty.push_back(_t.y0)
            tv.push_back(_t.v)
            th.push_back(_t.psi)
        self.thisptr.init(dp, dv, tx, ty, tv, th)


    def run(self, seq):
        cdef vector[int] _seq
        for _s in seq: _seq.push_back(_s)
        return self.thisptr.run(_seq)

    def run_all(self):
        cdef vector[int] _best_seq
        best_dur = self.thisptr.run_exhaustive(_best_seq)
        return best_dur, [_s for _s in _best_seq]
        
    def debug(self):
        cdef vector[float] psis
        psis = self.thisptr.get_psis()
        return [psi for psi in psis]

    def intercept_sequence_copy(self, drone, seq):
        drone = copy.deepcopy(drone)
        dt = self.run([_s.name-1 for _s in seq])
        drone.ts.append(dt)
        return drone, dt


