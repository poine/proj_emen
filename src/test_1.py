#! /usr/bin/env python3
#-*- coding: utf-8 -*-
import sys
import numpy as np, scipy.optimize, matplotlib.pyplot as plt

import pdb

import proj_manen as pm


def plot(ax, _d, _t, dt):
    pd0, pd1, pd2 = _d.get_pos(0), _d.get_pos(dt), _d.get_pos(dt+1.)
    vd0x, vd0y = _d.get_vel_e(0)
    vd0, psid0 = _d.get_vel(0)
    ax.plot(pd0[0], pd0[1], 'o', label=f'drone ${np.rad2deg(psid0):.1f} deg {vd0:.1f} m/s$')
    ax.arrow(pd0[0], pd0[1],  vd0x, vd0y, head_width=_d.v*0.025, head_length=_d.v*0.05, length_includes_head=True)
    ax.plot((pd0[0], pd2[0]), (pd0[1], pd2[1]), 'k--')
    
    pc0, pc1 = _t.get_pos(0), _t.get_pos(dt)
    ax.plot(pc0[0], pc0[1], 'o', label=f'cible ${np.rad2deg(_t.psi):.1f} deg {_t.v:.1f} m/s$')
    ax.arrow(pc0[0], pc0[1], _t.vx, _t.vy, head_width=_t.v*0.025, head_length=_t.v*0.05, length_includes_head=True)
    ax.plot((pc0[0], pc1[0]), (pc0[1], pc1[1]), 'k--')

    ax.plot(pc1[0], pc1[1], 'o', label=f'interception {dt:.2f} s')
    ax.axis('equal');ax.legend();ax.grid()
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)

def _check(drone, target, dt):
    pd, pt = drone.get_pos(dt), target.get_pos(dt)
    if not np.allclose(pd, pt):
        print(f'##### Failed: {pd} vs {pt}')
    #else:
    #    print(f'success: {pd} vs {pt}')

    
cases = [
    (5, 5, 10, np.deg2rad(10)), (-5, 5, 10, np.deg2rad(10)), (-5, -5, 10, np.deg2rad(10)), (5, -5, 10, np.deg2rad(10)),
    (0, 10, 10, np.deg2rad(-90)), (0, 10, 10, np.deg2rad(0)), (0, 10, 10, np.deg2rad(90)), (0, 10, 10, np.deg2rad(170)),
    (0, 10, 5, np.deg2rad(20)), (0, 10, 10, np.deg2rad(20)), (0, 10, 12, np.deg2rad(20)), (0, 10, 14, np.deg2rad(20))]

def test_set(_nc=4):
    fig = plt.figure(tight_layout=True, figsize=[12.8, 9.6]); fig.canvas.set_window_title('Interceptions')
    axes = fig.subplots(3,_nc, sharex=True)
    for _i, (xc0, xy0, vc, psic) in enumerate(cases):
        drone = pm.Drone(p0=[0., 0.], v0=15., h0=0.)
        target = pm.Actor(xc0, xy0, vc, psic, f'target_{_i}')
        psi, dt = pm.solve_1(drone, target)
        drone.add_leg(dt, psi)
        _check(drone, target, dt)
        plot(axes[np.divmod(_i, _nc)], drone, target, dt)
    plt.savefig('single_interception_examples.png')
    plt.show()

    
def test1(idx):
    drone = pm.Drone(p0=[0., 0.], v0=15., h0=0.)
    #xc0, xy0, vc, psic = 0, 10, 10, np.deg2rad(-90)
    xc0, xy0, vc, psic = cases[idx]
    target = pm.Actor(xc0, xy0, vc, psic, 'target')
    psi, dt = pm.solve_1(drone, target)
    drone.add_leg(dt, psi)
    _check(drone, target, dt)
    plot(plt.gca(), drone, target, dt)
    plt.show()

import time
def profile(idx=0, nloop=int(1e4)):
    drone = pm.Drone(p0=[0., 0.], v0=15., h0=0.)
    x0, y0, v, psi = cases[idx]
    target = pm.Actor(x0, x0, v, psi, 'target')
    _start = time.perf_counter()
    for i in range(nloop): 
        pm.solve_1(drone, target)
    _end = time.perf_counter()
    dt = _end-_start; ips=nloop/dt
    print(f'{nloop} iterations in {dt:.1f} s, {ips:.0f} iteration/s')


test_set()
#test1(int(sys.argv[1]))#5)
#profile()
