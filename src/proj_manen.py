#! /usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np, yaml
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
    
def solve_1(drone, _t): # a cos(psi) + b sin(psi) = c
    delta_p0 = _t.get_pos(drone.ts[-1])-drone.Xs[-1]
    a, b = delta_p0[1]*drone.v, -delta_p0[0]*drone.v
    c = delta_p0[1]*_t.vx-delta_p0[0]*_t.vy
    psis = 2*np.arctan(np.roots([a+c, -2*b, c-a]))
    delta_v = _to_eucl(drone.v, psis[0]) - _t._v
    if np.dot(delta_v, delta_p0) >= 0:
        psi = psis[0]
    else:
        delta_v = _to_eucl(drone.v, psis[1]) - _t._v
        psi = psis[1]
    dt = np.linalg.norm(delta_p0)/np.linalg.norm(delta_v)
    return psi, dt
        
def solve_sequence(drone, targets):
    for _targ in targets:
        psi, dt = solve_1(drone, _targ)
        drone.add_leg(dt, psi)
    return drone.ts[-1]

def load_scenario(filename):
    with open(filename) as f:
        _dict = yaml.load(f)
    targets = []
    for _k, _args in _dict.items():
        p0, v, h = _args['p0'], _args['v'], np.deg2rad(_args['h'])
        if _k.startswith('target'):
            targets.append(Actor(p0[0], p0[1], v, h, _k))
        else:
            drone = Drone(p0, v, h)
    return drone, targets

def make_scenario(ntarg=10, dp0=(0,0), dv=15):
    drone = Drone(dp0, dv, 0)
    ps = np.random.uniform(low=-20, high=20, size=(ntarg,2))
    hs = np.random.uniform(low=-np.pi, high=np.pi, size=ntarg)
    vs = np.random.uniform(low=0.1, high=12, size=ntarg)
    targets = [Actor(p0[0], p0[1], v, h, _k) for _k, (p0, h, v) in enumerate(zip(ps, hs, vs))]
    return drone, targets

def main():
    #drone, targets = make_scenario()
    drone, targets = load_scenario('./scenario_1.yaml')
    solve_sequence(drone, targets)
    print(f'duration: {drone.ts[-1]:.2f}s heading: {np.rad2deg(drone.psis[-1]):.1f} deg')
    

if __name__ == '__main__':
    main()
