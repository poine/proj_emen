import numpy as np, yaml, copy, itertools

import proj_manen as pm

import pdb

def plot_actor(ax, _a, dt=10., _name=''):
    p0, p1, vx, vy, v = _a.get_pos(0), _a.get_pos(dt), _a.vx, _a.vy, _a.v
    ax.plot(p0[0], p0[1], 'o', label=f'{_a.name}')#, label=f'drone {np.rad2deg(_d.psi):.1f} deg {_d.v:.1f} m/s')
    ax.arrow(p0[0], p0[1], vx, vy, head_width=v*0.025, head_length=v*0.05, length_includes_head=True)
    #ax.plot((p0[0], p1[0]), (p0[1], p1[1]), 'k--')
    
def plot_2d(ax, drone, targets, dt):
    plot_actor(ax, drone, dt=10., _name='drone')
    for _t in targets:
        plot_actor(ax, _t, dt=10., _name='t')
    ax.axis('equal')
    ax.legend()
    ax.grid()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)


def search_exhaustive(drone, targets, keep_nbest=1):
    perms = set(itertools.permutations(targets))
    best_dur, best_drones, best_targets = float('inf'), [], []
    for targets in perms:
        _drone = copy.deepcopy(drone)
        dur = pm.intercept_sequence(_drone, targets)
        if dur < best_dur:
            best_dur, best_targets, best_drones = dur, targets, _drone
    return best_drones, best_targets
    

def make_random_scenario(ntarg=10, dp0=(0,0), dv=15, plim=50):
    drone = pm.Drone(dp0, dv, 0)
    ps = np.random.uniform(low=-plim, high=plim, size=(ntarg,2))
    hs = np.random.uniform(low=-np.pi, high=np.pi, size=ntarg)
    vs = np.random.uniform(low=0.1, high=10., size=ntarg)
    targets = [pm.Actor(p0[0], p0[1], v, h, _k) for _k, (p0, h, v) in enumerate(zip(ps, hs, vs))]
    return drone, targets
    
def make_conv_div_scenario(ntarg=10, dp0=(20,0), dv=15, tv_mean=5., tv_std=1.5, conv=False):
    drone = pm.Drone(dp0, dv, 0)
    alphas = np.linspace(-np.pi, np.pi, ntarg, endpoint=False)
    hs = alphas
    #vs = np.random.normal(loc=tv_mean, scale=tv_std, size=ntarg)
    vs = np.random.uniform(low=0.5, high=10, size=ntarg)
    if conv: # we start over a circle
        ps = 125.*np.vstack([np.cos(alphas+np.pi), np.sin(alphas+np.pi)]).T
    else: # we start at center
        ps = np.zeros((ntarg, 2))
    targets = [pm.Actor(p0[0], p0[1], v, h, _k) for _k, (p0, h, v) in enumerate(zip(ps, hs, vs))]
    return drone, targets

def load_scenario(filename):
    with open(filename) as f:
        _dict = yaml.load(f)
    targets = []
    for _k, _args in _dict.items():
        p0, v, h = _args['p0'], _args['v'], np.deg2rad(_args['h'])
        if _k.startswith('target'):
            targets.append(pm.Actor(p0[0], p0[1], v, h, _k))
        else:
            drone = pm.Drone(p0, v, h)
    return drone, targets


