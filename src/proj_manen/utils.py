import os
import numpy as np, matplotlib.pyplot as plt, yaml, copy, itertools

import proj_manen as pm

import pdb
# normalize angles between -pi and pi
def norm_angles_mpi_pi(_a): return( _a + np.pi) % (2 * np.pi ) - np.pi
#
# Project directories
#
# we assume this file is proj_dir/src/proj_manen_utils.py
def proj_dir():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(dirname, '../..'))

def ressource(_r): return os.path.join(proj_dir(), _r)

# Getting tired of typing so many zeros
def parse_with_prefix(_s):
    try:
        res = int(float(_s))
    except ValueError:
        if   _s.endswith('k') or _s.endswith('K'): res = int(float(_s[:-1])*1e3)
        elif _s.endswith('m') or _s.endswith('M'): res = int(float(_s[:-1])*1e6)
        elif _s.endswith('g') or _s.endswith('G'): res = int(float(_s[:-1])*1e9)
    return res

# formating
def mmmm(_v): return np.min(_v), np.max(_v), np.mean(_v), np.median(_v)
def fmt_mmmm(_v):
    with np.printoptions(precision=2, suppress=True):
        return 'min/max: {} / {}  mean: {} med: {}'.format(*[np.array2string(__v) for __v in mmmm(_v)])

def format_seq(_s): return '-'.join([f'{int(_t.name):02d}' for _t in _s]) 

#
# Some plotting
#
def decorate(ax, title=None, xlab=None, ylab=None, legend=None, xlim=None, ylim=None, min_yspan=None):
    ax.xaxis.grid(color='k', linestyle='-', linewidth=0.2)
    ax.yaxis.grid(color='k', linestyle='-', linewidth=0.2)
    if xlab: ax.xaxis.set_label_text(xlab)
    if ylab: ax.yaxis.set_label_text(ylab)
    if title: ax.set_title(title, {'fontsize': 15 })
    if legend is not None:
        if legend == True: ax.legend(loc='best')
        else: ax.legend(legend, loc='best')
    if xlim is not None: ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
    if min_yspan is not None: ensure_yspan(ax, min_yspan)


def plot_scenario(scen, title='', annotate=False, fig=None, ax=None):
    if fig is None: fig = plt.gcf()
    if ax is None: ax = plt.gca()
    d0 = scen.drone.get_pos(0)
    ax.plot(d0[0], d0[1], 'X', markersize=10)
    for _t in scen.targets:
        plot_actor(ax, _t, dt=10., _name='', annotate=annotate)
    decorate(ax, title); ax.axis('equal')   

# plot one solution for several scenarios
def plot_scenarios(scens, names, sol_name=None):
    _n = len(names)
    fig = plt.figure(tight_layout=True, figsize=[5.12*_n, 5.12])
    axes = fig.subplots(1,_n)#, sharex=True)
    if _n==1: axes=[axes]
    for _s, _n, _ax in zip(scens, names, axes):
        if sol_name is not None:
            plot_solution(fig, _ax, _s, sol_name)
        else:
            plot_scenario(_s, _n, fig, _ax)
        
    
def plot_actor(ax, _a, dt=10., _name='', annotate=True):
    p0, p1, vx, vy, v = _a.get_pos(0), _a.get_pos(dt), _a.vx, _a.vy, _a.v
    _dot, = ax.plot(p0[0], p0[1], 'o', label=f'{_a.name}')#, label=f'drone {np.rad2deg(_d.psi):.1f} deg {_d.v:.1f} m/s')
    ax.plot(p1[0], p1[1], 'o', color=_dot.get_color(), markersize=4)
    #ax.arrow(p0[0], p0[1], vx, vy, color=_dot.get_color(), head_width=v*0.025, head_length=v*0.05, length_includes_head=True)
    ax.plot((p0[0], p1[0]), (p0[1], p1[1]), '--', color=_dot.get_color())
    if annotate: ax.annotate(_a.name, p0, p0+(0, 0.5))

# plot a serie of solutions on a single scenario
def plot_2d(fig, axes, scen, sol_names):
    for sol_name, ax in zip(sol_names, axes):
        plot_solution(fig, ax, scen, sol_name)

# plot one solution for one scenario
def plot_solution(fig, ax, scen, sol_name):
    seq = scen.solution_by_name(sol_name)[2]
    drone, dur = pm.intercept_sequence_copy(scen.drone, seq)
    #print(f'recomputed {format_seq(seq)} {dur:.2f} ')
    drone_poss = np.asarray(drone.Xs)
    ax.plot(drone_poss[:,0], drone_poss[:,1], '-X')
    for _target, _t in zip(seq, drone.ts[1:]):
        plot_actor(ax, _target, dt=_t)
    decorate(ax, f'{scen.name}/{sol_name} ({dur:.2f} s)'); ax.axis('equal')
    

        
# plot several solution for a single scenario
def plot_solutions(scen, names, filename=''):
    _n = len(names)
    fig = plt.figure(tight_layout=True, figsize=[5.12*_n, 5.12])
    fig.canvas.set_window_title(filename)
    axes = fig.subplots(1,_n, sharex=True)
    if _n==1: axes=[axes]
    return plot_2d(fig, axes, scen, names)
    
#
#
#
#import sortedcontainers  ... I'd want something like that...
def search_exhaustive(drone, targets, keep_all=False, display=False):
    perms = set(itertools.permutations(targets))
    if display: print(f'exhaustive search for {len(targets)} targets ({len(perms)} sequences)')
    best_dur, best_drone, best_targets, all_drones, all_targets= float('inf'), None, None, [],[]
    for _seq in perms:
        _drone, _dur = pm.intercept_sequence_copy(drone, _seq)
        if _dur < best_dur:
            best_dur, best_drone, best_targets = _dur, _drone, _seq
        if keep_all:
            all_drones.append(_drone); all_targets.append(_seq)
    if display: print(f'optimal seq {best_dur:.02f}s {format_seq(best_targets)}')
    return (all_drones, all_targets) if keep_all else (best_drone, best_targets)

# don't evaluate too long sequences till the end (with exceptions)
def search_exhaustive_improved(drone, targets, display=False):
    perms = set(itertools.permutations(targets))
    d0, seq0 = search_heuristic_closest(drone, targets)
    print(f'exhaustive_improved search for {len(targets)} targets ({len(perms)} sequences, heuristic {d0.flight_duration():.2f}s)')
    best_dur, best_drone, best_targets = d0.flight_duration(), None, None
    for targets in perms:
        _drone = copy.deepcopy(drone)
        try:
            dur = pm.intercept_sequence_if_shorter(_drone, targets, best_dur)
            if dur < best_dur:
                best_dur, best_drone, best_targets = dur, _drone, targets
        except pm.TimeExceededException: pass
    return best_drone, best_targets

# same without exceptions
def search_exhaustive_improved2(drone, targets, display=False):
    perms = set(itertools.permutations(targets))
    d0, seq0 = search_heuristic_closest(drone, targets)
    print(f'exhaustive_improved2 search for {len(targets)} targets ({len(perms)} sequences, heuristic {d0.flight_duration():.2f}s)')
    best_dur, best_drone, best_targets = d0.flight_duration(), None, None
    for targets in perms:
        _drone = copy.deepcopy(drone)
        dur = pm.intercept_sequence_if_shorter2(_drone, targets, best_dur)
        if dur is not None and dur < best_dur:
                best_dur, best_drone, best_targets = dur, _drone, targets
    return best_drone, best_targets



def search_heuristic_closest(drone, targets):
    drone = copy.deepcopy(drone) # we don't change our input arguments
    remaining, solution = targets.copy(), []
    while len(remaining) > 0:
        now = drone.flight_duration()
        rel_tpos = [_targ.get_pos(now)-drone.get_pos(now) for _targ in remaining]
        closest_target = remaining[np.argmin(np.linalg.norm(rel_tpos, axis=1))]
        remaining.remove(closest_target);solution.append(closest_target)
        psi, dt = pm.intercept_1(drone, closest_target)
        drone.add_leg(dt, psi)
    return drone, solution

