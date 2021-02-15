# distutils: language = c++
#
#
#  cython interface to the C++ target interception  code 
#
# Author: Poine-2021
#
#
# cython: language_level=3
#

import numpy as np, copy

from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "pm/pm.h":
    ctypedef float PmType
    cdef cppclass c_Solver "Solver":
        c_Solver()
        bool init(PmType* dp, float dv, vector[PmType] tx, vector[PmType] ty, vector[float] tv, vector[float] th)
        PmType search_sa(vector[int] start_seq, unsigned int nepoch, float T0, vector[int] best_seq, int display)
        PmType search_exhaustive(vector[int] &best_seq)
        PmType run_sequence(vector[int] seq)
        PmType run_sequence_threshold(vector[int] seq, PmType max_t)
        PmType run_sequence_random(vector[int] &best_seq)
        vector[float] get_psis()

cdef class Solver:
    cdef c_Solver *thisptr

    def __cinit__(self, drone=None, targets=None):
        self.thisptr = new c_Solver()
        if drone is not None and targets is not None:
            self.init(drone, targets)
        
    def init(self, drone, targets):
        cdef PmType dp[2]
        cdef float dv
        dp[0] = drone.p0[0]
        dp[1] = drone.p0[1]
        dv = drone.v
        cdef vector[PmType] tx
        cdef vector[PmType] ty
        cdef vector[float] tv
        cdef vector[float] th
        for _t in targets:
            tx.push_back(_t.x0)
            ty.push_back(_t.y0)
            tv.push_back(_t.v)
            th.push_back(_t.psi)
        self.thisptr.init(dp, dv, tx, ty, tv, th)

    def search_sa(self, start_seq, nepoch, T0=1., display=0):
        cdef vector[int] _start_seq
        for _s in start_seq: _start_seq.push_back(_s)
        cdef vector[int] _best_seq
        best_dur = self.thisptr.search_sa(_start_seq, nepoch, T0, _best_seq, display)
        return best_dur, [_s for _s in _best_seq]
        
    def search_exhaustive(self):
        cdef vector[int] _best_seq
        best_dur = self.thisptr.search_exhaustive(_best_seq)
        return best_dur, [_s for _s in _best_seq]

    def run_sequence(self, seq):
        cdef vector[int] _seq
        for _s in seq: _seq.push_back(_s)
        return self.thisptr.run_sequence(_seq)

    def run_sequence_threshold(self, seq, max_t):
        cdef vector[int] _seq
        for _s in seq: _seq.push_back(_s)
        return self.thisptr.run_sequence_threshold(_seq, max_t)

    def run_sequence_random(self):
        cdef vector[int] _seq
        _dur = self.thisptr.run_sequence_random(_seq)
        return _dur, [_s for _s in _seq]
        
    def debug(self):
        cdef vector[float] psis
        psis = self.thisptr.get_psis()
        return [psi for psi in psis]

    # mimic the original python prototype
    def intercept_sequence_copy(self, drone, seq):
        drone = copy.deepcopy(drone)
        dt = self.run_sequence([_s.name-1 for _s in seq])
        drone.ts.append(dt)
        return drone, dt

    def intercept_sequence_copy_threshold(self, drone, seq, max_t):
        drone = copy.deepcopy(drone)
        dt = self.run_sequence_threshold([_s.name-1 for _s in seq], max_t)
        drone.ts.append(dt)
        return drone, dt


