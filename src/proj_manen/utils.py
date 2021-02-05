import os
import numpy as np, matplotlib.pyplot as plt, yaml, copy, itertools

import proj_manen as pm

import pdb
def norm_angles_mpi_pi(_a): return( _a + np.pi) % (2 * np.pi ) - np.pi
#
# Project directories
#
# we assume this file is proj_dir/src/proj_manen_utils.py
def proj_dir():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(dirname, '../..'))

def ressource(_r): return os.path.join(proj_dir(), _r)

#
# Some plotting
#
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


def plot_scenario(scen, name='', fig=None, ax=None):
    if fig is None: fig = plt.gcf()
    if ax is None: ax = plt.gca()
    d0 = scen.drone.get_pos(0)
    ax.plot(d0[0], d0[1], 'X', markersize=10)
    for _t in scen.targets:
        plot_actor(ax, _t, dt=10., _name='')
    decorate(ax, f'{name}');ax.axis('equal')   

def plot_scenarios(scens, names, sol_name=None):
    _n = len(names)
    fig = plt.figure(tight_layout=True, figsize=[5.12*_n, 5.12])
    axes = fig.subplots(1,_n)#, sharex=True)
    if _n==1: axes=[axes]
    for _s, _n, _ax in zip(scens, names, axes):
        if sol_name is not None:
            plot_2d(fig, [_ax], _s, sol_name)
        else:
            plot_scenario(_s, _n, fig, _ax)
        
    
def plot_actor(ax, _a, dt=10., _name='', annotate=True):
    p0, p1, vx, vy, v = _a.get_pos(0), _a.get_pos(dt), _a.vx, _a.vy, _a.v
    _dot, = ax.plot(p0[0], p0[1], 'o', label=f'{_a.name}')#, label=f'drone {np.rad2deg(_d.psi):.1f} deg {_d.v:.1f} m/s')
    ax.plot(p1[0], p1[1], 'o', color=_dot.get_color(), markersize=4)
    #ax.arrow(p0[0], p0[1], vx, vy, color=_dot.get_color(), head_width=v*0.025, head_length=v*0.05, length_includes_head=True)
    ax.plot((p0[0], p1[0]), (p0[1], p1[1]), '--', color=_dot.get_color())
    if annotate: ax.annotate(_a.name, p0, p0+(0, 0.5))
    
def plot_2d(fig, axes, scen, names):
    for name, ax in zip(names, axes):
        seq = scen.solution_by_name(name)[2]
        drone, dur = pm.intercept_sequence_copy(scen.drone, seq)
        #print(f'recomputed {format_seq(seq)} {dur:.2f} ')
        drone_poss = np.asarray(drone.Xs)
        ax.plot(drone_poss[:,0], drone_poss[:,1], '-X')
        for _target, _t in zip(seq, drone.ts[1:]):
            plot_actor(ax, _target, dt=_t)
        decorate(ax, f'{name}: {dur:.2f} s'); ax.axis('equal')

# plot several solution for a single scenario
def plot_solutions(scen, names, filename=''):
    _n = len(names)
    fig = plt.figure(tight_layout=True, figsize=[5.12*_n, 5.12])
    fig.canvas.set_window_title(filename)
    axes = fig.subplots(1,_n, sharex=True)
    if _n==1: axes=[axes]
    return plot_2d(fig, axes, scen, names)
    
#
# Scenarios
#
def make_random_scenario(ntarg, dp0=(10,0), dv=15,
                          tp={'kind':'uniform', 'low':-50, 'high':50},
                          th={'kind':'uniform', 'low':-np.pi, 'high':np.pi},
                          tv={'kind':'normal', 'mean':5., 'std':0.5}):
    plim = 50.
    drone = pm.Drone(dp0, dv, 0)

    alphas = np.linspace(-np.pi, np.pi, ntarg, endpoint=False)
    # Positions
    if tp['kind'] == 'point':
        ps = np.zeros((ntarg, 2))
    elif tp['kind'] == 'uniform':
        ps = np.random.uniform(low=tp['low'], high=tp['high'], size=(ntarg,2))
    elif tp['kind'] == 'circle':
        ps = tp['r']*np.vstack([np.cos(alphas+np.pi), np.sin(alphas+np.pi)]).T
    elif tp['kind'] == 'line':
        ps = np.vstack([-75*np.ones(ntarg), np.linspace(-tp['len'], tp['len'], ntarg)]).T
    elif tp['kind'] == 'grid':
        ps = tp['d']*np.array([np.divmod(i, tp['nr']) for i in range(ntarg)])

    # Headings
    if th['kind'] == 'uniform':
        hs = np.random.uniform(low=th['low'], high=th['high'], size=ntarg)
    elif th['kind'] == 'normal':
        hs = np.random.normal(loc=th['mean'], scale=th['std'], size=ntarg)
    elif th['kind'] == 'toward':
        hs = norm_angles_mpi_pi(alphas + np.random.normal(loc=th['mean'], scale=th['std'], size=ntarg))
    elif th['kind'] == 'away':
        hs = norm_angles_mpi_pi(alphas + np.pi+ np.random.normal(loc=th['mean'], scale=th['std'], size=ntarg))
    # Velocities
    if tv['kind'] == 'normal':
        vs = np.random.normal(loc=tv['mean'], scale=tv['std'], size=ntarg)
    elif tv['kind'] == 'uniform':
        vs = np.random.uniform(low=tv['low'], high=tv['high'], size=ntarg)

    targets = [pm.Actor(p0[0], p0[1], v, h, f'{_k+1}') for _k, (p0, h, v) in enumerate(zip(ps, hs, vs))]
    return drone, targets



def load_scenario(filename):
    print(f'loading scenario from file: {filename}')
    with open(filename) as f:
        _dict = yaml.load(f)
    targets, solutions = [], []
    for _k, _args in _dict.items():
        if _k == 'targets':
            for __k, __args in _dict[_k].items():
                p0, v, h = __args['p0'], __args['v'], np.deg2rad(__args['h'])
                targets.append(pm.Actor(p0[0], p0[1], v, h, __k))
        elif _k == 'drone':
            p0, v, h = _args['p0'], _args['v'], np.deg2rad(_args['h'])
            drone = pm.Drone(p0, v, h)
        elif _k == 'solutions':
            for __k, __args in _dict[_k].items():
                solutions.append([__k, __args['duration'], __args['seq']])
    _ta_by_name = {_t.name:_t for _t in targets}
    for _s in solutions: # replace list of target names with references to actual targets
        _s[2] = [_ta_by_name[_tn] for _tn in _s[2]]
    print(f'  {len(targets)} targets, {len(solutions)} known solutions')
    for _s in solutions: print(f'     {_s[0]:12s}: {_s[1]: 8.2f} s     {format_seq(_s[2])}')
    return drone, targets, solutions

def save_scenario(filename, drone, targets, solutions=[]):
    print(f'saving scenario to: {filename}')
    txt = 'targets:\n'
    for _t in targets:
        txt += f'  {_t.name}:\n    p0: [{_t.x0},{_t.y0}]\n    v: {_t.v}\n    h: {np.rad2deg(_t.psi)}\n\n'
    _d = drone    
    txt += f'{_d.name}:\n  p0: [{_d.x0},{_d.y0}]\n  v: {_d.v}\n  h: {np.rad2deg(_d.psi)}\n\n'
    if len(solutions)>0:
        txt += 'solutions:\n'
        for _name, _dur, _seq in solutions:
            txt+=f'  {_name}:\n    duration: {_dur}\n    seq: {[_t.name for _t in _seq]}\n'
    with open(filename, 'w') as f:
        f.write(txt)


#def format_seq(_s): tmp = [int(_t.name.split('_')[-1]) for _t in _s]; return '-'.join([f'{_f:02d}' for _f in tmp]) 
def format_seq(_s): return '-'.join([f'{int(_t.name):02d}' for _t in _s]) 

class Scenario:
    def __init__(self, **kwargs):
        if 'filename' in kwargs: self.load(kwargs['filename'])
    
    def load(self, filename):
       self.drone, self.targets, self.solutions = load_scenario(filename)
       self.solutions_by_name = {_s[0]: _s for _s in self.solutions}

    def save(self, filename):
        save_scenario(filename, self.drone, self.targets, self.solutions)

    def add_solution(self, name, dur, seq):
        self.solutions.append([name, dur, seq])
        self.solutions_by_name[name] = self.solutions[-1]

    def nb_solutions(self): return len(self.solutions)
    
    def solution_by_name(self, name):
        return self.solutions_by_name[name]



    def fix(self):
        sols_by_seq = {}
        print(f'all solutions: {[_s[0] for _s in self.solutions]}')
        for _s  in self.solutions:
            _name, _dur, _seq = _s
            str_seq = format_seq(_seq)
            #print(str_seq)
            #pdb.set_trace()
            if str_seq in sols_by_seq:
                print(f'dupe {_name} was {sols_by_seq[str_seq][0]}')
            else:
                print(f'unique {_name}')
                sols_by_seq[str_seq] = _s

    



class ScenarioFactory:
    _def_tv = {'kind':'normal', 'mean':5., 'std':0.}
    
    scenarios = {
        71:  ['scenario_7_1.yaml', # defaults: uniform law for pos and heading, normal law for speed
             lambda: make_random_scenario(ntarg=7)],
        72:  ['scenario_7_2.yaml', # circle toward
             lambda: make_random_scenario(ntarg=7, dp0=(30,5), dv=15,
                                           tp={'kind':'circle', 'r':75}, th={'kind':'toward', 'mean':np.deg2rad(0.), 'std':np.deg2rad(10.)})],
        73:  ['scenario_7_3.yaml', # circle away
             lambda: make_random_scenario(ntarg=7, dp0=(10,0), dv=15,
                                           tp={'kind':'circle', 'r':25}, th={'kind':'away', 'mean':np.deg2rad(0.), 'std':np.deg2rad(10.)})],
        74:  ['scenario_7_4.yaml', # line
             lambda: make_random_scenario(ntarg=7, dp0=(30,5), dv=15,
                                           tp={'kind':'line', 'len':50}, th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(10.)})],
        78:  ['scenario_7_8.yaml', # grid 1
             lambda: make_random_scenario(ntarg=7, dp0=(0,-10), dv=15,
                                           tp={'kind':'grid', 'nr':3, 'd':15},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        79:  ['scenario_7_9.yaml', # grid 2
             lambda: make_random_scenario(ntarg=7, dp0=(0,-10), dv=15,
                                           tp={'kind':'grid', 'nr':3, 'd':15},
                                           th={'kind':'normal', 'mean':np.deg2rad(30.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],

        81:  ['scenario_8_1.yaml', # 8 targets, defaults: uniform law for pos and heading, normal law for speed
             lambda: make_random_scenario(ntarg=8)],
        88:  ['scenario_8_8.yaml', # grid 1
             lambda: make_random_scenario(ntarg=8, dp0=(-10,-10), dv=15,
                                           tp={'kind':'grid', 'nr':3, 'd':15},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        89:  ['scenario_8_9.yaml', # grid 2
             lambda: make_random_scenario(ntarg=8, dp0=(-10,-10), dv=15,
                                           tp={'kind':'grid', 'nr':3, 'd':15},
                                           th={'kind':'normal', 'mean':np.deg2rad(20.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        
        91:  ['scenario_9_1.yaml', # 9 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=9)],
        92:  ['scenario_9_2.yaml', # circle toward
              lambda: make_random_scenario(ntarg=9, dp0=(0,0), dv=15,
                                           tp={'kind':'circle', 'r':75},
                                           th={'kind':'toward', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        93:  ['scenario_9_3.yaml', # circle away
              lambda: make_random_scenario(ntarg=9, dp0=(10,0), dv=15,
                                           tp={'kind':'circle', 'r':25}, th={'kind':'away', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)})],
        94:  ['scenario_9_4.yaml', # line
             lambda: make_random_scenario(ntarg=9, dp0=(30,5), dv=15,
                                           tp={'kind':'line', 'len':50}, th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)})],
        98:  ['scenario_9_8.yaml', # grid 1
             lambda: make_random_scenario(ntarg=9, dp0=(-10,-10), dv=15,
                                           tp={'kind':'grid', 'nr':3, 'd':15},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        99:  ['scenario_9_9.yaml', # grid 2
              lambda: make_random_scenario(ntarg=9, dp0=(-10,-10), dv=15,
                                           tp={'kind':'grid', 'nr':3, 'd':15},
                                           th={'kind':'normal', 'mean':np.deg2rad(30.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],

        
        101: ['scenario_10_1.yaml', # 10 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=10)],
        108: ['scenario_10_8.yaml', # grid 1
              lambda: make_random_scenario(ntarg=10, dp0=(-10,-10), dv=15,
                                            tp={'kind':'grid', 'nr':3, 'd':15},
                                            th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        109: ['scenario_10_9.yaml', # grid 2
              lambda: make_random_scenario(ntarg=10, dp0=(-10,-10), dv=15,
                                            tp={'kind':'grid', 'nr':3, 'd':15},
                                            th={'kind':'normal', 'mean':np.deg2rad(20.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],

        151: ['scenario_15_1.yaml', # 15 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=15)],
        152: ['scenario_15_2.yaml', # circle toward
              lambda: make_random_scenario(ntarg=15, dp0=(30,5), dv=15,
                                           tp={'kind':'circle', 'r':75},
                                           th={'kind':'toward', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        153: ['scenario_15_3.yaml', # circle away
              lambda: make_random_scenario(ntarg=15, dp0=(10,0), dv=15,
                                           tp={'kind':'circle', 'r':25},
                                           th={'kind':'away', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        154: ['scenario_15_4.yaml', # line
              lambda: make_random_scenario(ntarg=15, dp0=(30,5), dv=15,
                                           tp={'kind':'line', 'len':50}, th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(10.)})],
        155: ['scenario_15_5.yaml', # circle headings normal
             lambda: make_random_scenario(ntarg=15, dp0=(30,5), dv=15,
                                           tp={'kind':'circle', 'r':75}, th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(10.)})],
        156: ['scenario_15_6.yaml', # circle headings normal
             lambda: make_random_scenario(ntarg=15, dp0=(25,25), dv=15,
                                           tp={'kind':'circle', 'r':75},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],
        157: ['scenario_15_7.yaml', # circle toward
             lambda: make_random_scenario(ntarg=15, dp0=(30,5), dv=15,
                                           tp={'kind':'circle', 'r':75},
                                           th={'kind':'toward', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],
        158: ['scenario_15_8.yaml', # circle away
             lambda: make_random_scenario(ntarg=15, dp0=(0,0), dv=15,
                                           tp={'kind':'circle', 'r':15},
                                           th={'kind':'away', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],
        159: ['scenario_15_9.yaml', # grid
             lambda: make_random_scenario(ntarg=15, dp0=(-10,-10), dv=15,
                                           tp={'kind':'grid', 'nr':5, 'd':15},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],
        1510: ['scenario_15_10.yaml', # grid
             lambda: make_random_scenario(ntarg=15, dp0=(-10,-10), dv=15,
                                          tp={'kind':'grid', 'nr':5, 'd':15},
                                          th={'kind':'normal', 'mean':np.deg2rad(20.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],

        301: ['scenario_30_1.yaml', # 30 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=30, tp={'kind':'uniform', 'low':-100, 'high':100})],
        302: ['scenario_30_2.yaml', # circle toward
              lambda: make_random_scenario(ntarg=30, dp0=(0,0), dv=15,
                                           tp={'kind':'circle', 'r':200.},
                                           th={'kind':'toward', 'mean':np.deg2rad(0.), 'std':np.deg2rad(1.)}, tv=ScenarioFactory._def_tv)],
        303: ['scenario_30_3.yaml', # circle away
              lambda: make_random_scenario(ntarg=30, dp0=(10,0), dv=15,
                                           tp={'kind':'circle', 'r':25},
                                           th={'kind':'away', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        304: ['scenario_30_4.yaml', # line
              lambda: make_random_scenario(ntarg=30, dp0=(50,0), dv=15,
                                           tp={'kind':'line', 'len':150},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        306: ['scenario_30_6.yaml', # circle headings normal
             lambda: make_random_scenario(ntarg=30, dp0=(25,25), dv=15,
                                           tp={'kind':'circle', 'r':75},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],
        307: ['scenario_30_7.yaml', # circle headings normal, faster
             lambda: make_random_scenario(ntarg=30, dp0=(25,25), dv=15,
                                           tp={'kind':'circle', 'r':75},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':7.5, 'std':0.})],
        308: ['scenario_30_8.yaml', # grid 1
             lambda: make_random_scenario(ntarg=30, dp0=(-10,-10), dv=15,
                                           tp={'kind':'grid', 'nr':5, 'd':15},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],
        309: ['scenario_30_9.yaml', # grid 2
             lambda: make_random_scenario(ntarg=30, dp0=(-10,-10), dv=15,
                                           tp={'kind':'grid', 'nr':5, 'd':15},
                                           th={'kind':'normal', 'mean':np.deg2rad(25.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],
        50: ['scenario_60_1.yaml', # 60 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=60, tp={'kind':'uniform', 'low':-100, 'high':100})],
        602: ['scenario_60_2.yaml', # circle toward
              lambda: make_random_scenario(ntarg=60, dp0=(30,5), dv=15,
                                           tp={'kind':'circle', 'r':120},
                                           th={'kind':'toward', 'mean':np.deg2rad(0.), 'std':np.deg2rad(1.)}, tv=ScenarioFactory._def_tv)],
        603: ['scenario_60_3.yaml', # circle away
              lambda: make_random_scenario(ntarg=60, dp0=(0,0), dv=15,
                                           tp={'kind':'circle', 'r':25},
                                           th={'kind':'away', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        605: ['scenario_60_5.yaml', # 60 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=60, tp={'kind':'uniform', 'low':-250, 'high':250})],
        606: ['scenario_60_6.yaml', # circle headings normal
              lambda: make_random_scenario(ntarg=60, dp0=(25,25), dv=15,
                                           tp={'kind':'circle', 'r':100},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],
        6061: ['scenario_60_6_1.yaml', # circle headings normal with noise
              lambda: make_random_scenario(ntarg=60, dp0=(25,25), dv=15,
                                           tp={'kind':'circle', 'r':100},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(10.)}, tv={'kind':'normal', 'mean':5., 'std':1.5})],
        6062: ['scenario_60_6_2.yaml', # circle headings normal with less noise
              lambda: make_random_scenario(ntarg=60, dp0=(25,25), dv=15,
                                           tp={'kind':'circle', 'r':100},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(5.)}, tv={'kind':'normal', 'mean':5., 'std':0.5})],
        1205: ['scenario_120_5.yaml', # 120 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=120, tp={'kind':'uniform', 'low':-250, 'high':250})],
        1206: ['scenario_120_6.yaml', # circle headings normal
              lambda: make_random_scenario(ntarg=120, dp0=(25,25), dv=15,
                                           tp={'kind':'circle', 'r':200},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],
    }


    def filename(idx): return ScenarioFactory.scenarios[idx][0]
    
    def make(idx):
        _filename, _f = ScenarioFactory.scenarios[idx]
        drone, targets = _f()
        s = Scenario()
        s.drone, s.targets, s.solutions, s.solutions_by_name = drone, targets, [], {}
        #s.save(ScenarioFactory.filenames[idx])
        return s



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

