import numpy as np, yaml, copy, itertools

import proj_manen as pm

import pdb

def decorate(ax, title=None, xlab=None, ylab=None, legend=None, xlim=None, ylim=None, min_yspan=None):
    ax.xaxis.grid(color='k', linestyle='-', linewidth=0.2)
    ax.yaxis.grid(color='k', linestyle='-', linewidth=0.2)
    if xlab: ax.xaxis.set_label_text(xlab)
    if ylab: ax.yaxis.set_label_text(ylab)
    if title: ax.set_title(title, {'fontsize': 20 })
    if legend is not None:
        if legend == True: ax.legend(loc='best')
        else: ax.legend(legend, loc='best')
    if xlim is not None: ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None: ax.set_ylim(ylim[0], ylim[1])
    if min_yspan is not None: ensure_yspan(ax, min_yspan)

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


def make_random_scenario(ntarg=10, dp0=(0,0), dv=15, plim=50):
    drone = pm.Drone(dp0, dv, 0)
    ps = np.random.uniform(low=-plim, high=plim, size=(ntarg,2))
    hs = np.random.uniform(low=-np.pi, high=np.pi, size=ntarg)
    vs = np.random.uniform(low=0.1, high=10., size=ntarg)
    targets = [pm.Actor(p0[0], p0[1], v, h, f'target_{_k+1}') for _k, (p0, h, v) in enumerate(zip(ps, hs, vs))]
    return drone, targets
    
def make_conv_div_scenario(ntarg=10, dp0=(20,0), dv=15, tv_mean=5., tv_std=1.5, tv_uni=False, tv_low=0.5, tv_high=10., conv=False, other=False, tr=100.):
    drone = pm.Drone(dp0, dv, 0)
    alphas = np.linspace(-np.pi, np.pi, ntarg, endpoint=False)
    hs = alphas + np.random.normal(loc=0, scale=np.deg2rad(15), size=ntarg)
    if tv_uni:
        vs = np.random.uniform(low=tv_low, high=tv_high, size=ntarg)
    else:
        vs = np.random.normal(loc=tv_mean, scale=tv_std, size=ntarg)
    if conv: # we start over a circle
        ps = tr*np.vstack([np.cos(alphas+np.pi), np.sin(alphas+np.pi)]).T
    else: # we start at center
        ps = np.zeros((ntarg, 2))
    if other:
        #pdb.set_trace()
        ps = np.vstack([-75*np.ones(ntarg), np.linspace(-100, 100, ntarg)]).T
        hs = 0 + np.random.normal(loc=0, scale=np.deg2rad(15), size=ntarg)
        
    targets = [pm.Actor(p0[0], p0[1], v, h, f'target_{_k+1}') for _k, (p0, h, v) in enumerate(zip(ps, hs, vs))]
    return drone, targets


def load_scenario(filename):
    print(f'loading scenario from file: {filename}')
    with open(filename) as f:
        _dict = yaml.load(f)
    targets, solutions = [], []
    for _k, _args in _dict.items():
        if _k.startswith('target'):
            p0, v, h = _args['p0'], _args['v'], np.deg2rad(_args['h'])
            targets.append(pm.Actor(p0[0], p0[1], v, h, _k))
        elif _k == 'drone':
            p0, v, h = _args['p0'], _args['v'], np.deg2rad(_args['h'])
            drone = pm.Drone(p0, v, h)
        elif _k == 'solutions':
            for __k, __args in _dict[_k].items():
                solutions.append([__k, __args['duration'], __args['seq']])
    _ta_by_name = {_t.name:_t for _t in targets}
    for _s in solutions: # replace list of target names with references to actual targets
        _s[2] = [_ta_by_name[_tn] for _tn in _s[2]]
    return drone, targets, solutions

def save_scenario(filename, drone, targets, solutions=[]):
    print(f'saving scenario to: {filename}')
    txt = ''
    for _t in targets:
        txt += f'{_t.name}:\n  p0: [{_t.x0},{_t.y0}]\n  v: {_t.v}\n  h: {np.rad2deg(_t.psi)}\n\n'
    _d = drone    
    txt += f'{_d.name}:\n  p0: [{_d.x0},{_d.y0}]\n  v: {_d.v}\n  h: {np.rad2deg(_d.psi)}\n\n'
    if len(solutions)>0:
        txt += 'solutions:\n'
        for _name, _dur, _seq in solutions:
            txt+=f'  {_name}:\n    duration: {_dur}\n    seq: {_seq}\n'
    with open(filename, 'w') as f:
        f.write(txt)


def sol_by_name(solutions, name):
    for _name, _dur, _seq in solutions:
        if _name == name: return _dur, _seq
    return None, None
