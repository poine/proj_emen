#! /usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np, copy
import pdb

def _to_eucl(n, h): return n*np.array([np.cos(h), np.sin(h)])
class Actor: # a moving actor (target): cst speed, cts heading
    def __init__(self, x0, y0, v, psi, name):
        self.name = name
        self.x0, self.y0 = self.p0 = np.array([x0, y0])
        self.set_vel(v, psi)

    def set_vel(self, v, psi):
        self.v, self.psi = v, psi
        self.vx, self.vy = self._v = _to_eucl(self.v, psi)

    def get_pos(self, t): return self.p0 + self._v*t
    def get_vel_e(self, t): return self._v
    def get_vel(self, t): return self.v, self.psi

class Drone(Actor): # piecewise cst heading
    def __init__(self, p0, v0, h0):
        Actor.__init__(self, p0[0], p0[1], v0, h0, 'drone')
        self.clear_traj()

    def clear_traj(self):
        self.ts, self.psis, self.Xs = [0.],[],[self.p0]
        
    def add_leg(self, dt, psi):
        self.ts.append(self.ts[-1]+dt)
        self.psis.append(psi)
        self.Xs.append(self.Xs[-1]+_to_eucl(self.v, self.psis[-1])*dt)
            
    def get_pos(self, t):
        if len(self.ts) <= 1: return Actor.get_pos(self, t)
        i = self._idx_leg(t)
        return self.Xs[i]+self.get_vel_e_leg(i)*(t-self.ts[i])

    def get_vel_e_leg(self, i): return _to_eucl(self.v, self.psis[i])
    def get_vel_e(self, t):
        if len(self.ts) <= 1: return Actor.get_vel_e(self, t)
        return self.get_vel_e_leg(self._idx_leg(t))

    def get_vel_leg(self, i): return self.v, self.psis[i]
    def get_vel(self, t):
        if len(self.ts) <= 1: return Actor.get_vel(self, t)
        return self.get_vel_leg(self._idx_leg(t))
    def _idx_leg(self, t): return np.argmax(t<np.asarray(self.ts))-1
    def flight_duration(self, leg=-1): return self.ts[leg]
    
def intercept_1(drone, target): # a cos(psi) + b sin(psi) = c
    delta_p0 = drone.Xs[-1] - target.get_pos(drone.flight_duration())
    if np.linalg.norm(delta_p0) < 1e-12:
        print('###over target')
        return 0., 0.  # the drone is already over the target
    a, b = delta_p0[1]*drone.v, -delta_p0[0]*drone.v
    c = delta_p0[1]*target.vx-delta_p0[0]*target.vy
    if abs(a) < 1e-12: # we know b is not zero thanks to previous over_target test
        psi1 = np.arcsin(c/b); psi2 = np.pi - psi1
        delta_v1 = _to_eucl(drone.v, psi1) - target._v
        delta_v2 = _to_eucl(drone.v, psi2) - target._v
        if np.dot(delta_v1, delta_p0) <= 0:
            psi, delta_v = psi1, delta_v1
        else:
            psi, delta_v = psi2, delta_v2
        #print(f'{target.name} {psi1:.1f} {psi2:.1f} {delta_p0} {delta_v1} {delta_v2}')
        #pdb.set_trace()
    else:
        psis = 2*np.arctan(np.roots([a+c, -2*b, c-a]))
        #if len(psis)<1:
        #    print('no solution??? FIXME')
        #    pdb.set_trace()
        #if len(psis)<2:
        #    print('single root??? FIXME')
        #    pdb.set_trace()
        delta_v = _to_eucl(drone.v, psis[0]) - target._v
        if np.dot(delta_v, delta_p0) <= 0 or len(psis)<2:
            psi = psis[0]
        else:
            delta_v = _to_eucl(drone.v, psis[1]) - target._v
            psi = psis[1]
        #print(f'{target.name} {psis} {psi}')

    dt = np.linalg.norm(delta_p0)/np.linalg.norm(delta_v)
    return psi, dt
        
def intercept_sequence(drone, targets):
    for target in targets:
        psi, dt = intercept_1(drone, target)
        drone.add_leg(dt, psi)
    return drone.ts[-1]

def intercept_sequence_copy(drone, targets):
    drone = copy.deepcopy(drone)
    return drone, intercept_sequence(drone, targets)

# first version: read below
# class TimeExceededException(Exception):
#     pass
# def intercept_sequence_if_shorter(drone, targets, max_t):
#     for target in targets:
#         psi, dt = intercept_1(drone, target)
#         drone.add_leg(dt, psi)
#         if drone.flight_duration() >= max_t: raise TimeExceededException
#     return drone.ts[-1]
#
# def intercept_sequence_if_shorter2(drone, targets, max_t):
#     for target in targets:
#         psi, dt = intercept_1(drone, target)
#         drone.add_leg(dt, psi)
#         if drone.flight_duration() >= max_t: return None
#     return drone.ts[-1]


# Intercept a sequence up to a given time
# hope this has two nice effects: speeds up computations (maybe) and make us more resilient to overflows in time and positions
def intercept_sequence_copy_threshold(drone, targets, max_t):
    drone = copy.deepcopy(drone)
    for target in targets:
        psi, dt = intercept_1(drone, target)
        drone.add_leg(dt, psi)
        if drone.flight_duration() >= max_t: break
    return drone, drone.ts[-1]







def main():
    drone = Drone(p0=[0., 0.], v0=15., h0=0.)
    target1 = Actor(5., 5., 10, np.deg2rad(10.), 'target1')
    intercept_1(drone, target1)
    target2 = Actor(-5., 5., 10, np.deg2rad(-10.), 'target2')
    intercept_sequence(drone, [target1, target2])
    print(f'duration: {drone.ts[-1]:.2f}s heading: {np.rad2deg(drone.psis[-1]):.1f} deg')

if __name__ == '__main__':
    main()
