import os
import numpy as np, matplotlib.pyplot as plt, yaml, copy, itertools

import proj_manen as pm

import pdb
    
#
# Scenarios
#
def make_random_scenario(ntarg, dp0=(10,0), dv=15,
                          tp={'kind':'uniform', 'low':-50, 'high':50},
                          th={'kind':'uniform', 'low':-np.pi, 'high':np.pi},
                          tv={'kind':'normal', 'mean':5., 'std':0.5}):
    drone = pm.Drone(dp0, dv, 0)

    alphas = np.linspace(-np.pi, np.pi, ntarg, endpoint=False)
    # Positions
    if tp['kind'] == 'point':
        ps = np.zeros((ntarg, 2))
    elif tp['kind'] == 'uniform':
        ps = np.random.uniform(low=tp['low'], high=tp['high'], size=(ntarg,2))
    elif tp['kind'] == 'circle':
        (cx, cy), cr = tp.get('center', (0, 0)), tp['r']
        ps = np.vstack([cx+cr*np.cos(alphas+np.pi), cy+cr*np.sin(alphas+np.pi)]).T
    elif tp['kind'] == 'line':
        (cx,cy), gamma = tp.get('center', (-75, 0)), tp.get('gamma', np.pi/2)
        _l = np.linspace(-tp['len'], tp['len'], ntarg)
        ps = np.vstack([cx+np.cos(gamma)*_l, cy+np.sin(gamma)*_l]).T
    elif tp['kind'] == 'grid':
        _c = np.array(tp.get('center', (0, 0)))
        ps = tp['d']*np.array([np.divmod(i, tp['nr']) for i in range(ntarg)])+_c

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

def _print_solution(_s):
    #if len(targets) < 31: print(f'     {_s[0]:12s}: {_s[1]: 8.3f} s     {format_seq(_s[2])}')
    #else:  print(f'     {_s[0]:12s}: {_s[1]: 8.3f} s     {format_seq(_s[2][:30])}...')
    sol_txt = format_seqs(_s[2]) if type(_s[2][0]) == list else format_seq(_s[2])
    print(f'     {_s[0]:12s}: {_s[1]: 8.3f} s   {sol_txt}')

def load_scenario(filename):
    print(f'loading scenario from file: {filename}')
    with open(filename) as f:
        _dict = yaml.load(f, Loader=yaml.SafeLoader)
    drones, targets, solutions = [], [], []
    for _k, _args in _dict.items():
        if _k == 'targets':
            for __k, __args in _dict[_k].items():
                p0, v, h = __args['p0'], __args['v'], np.deg2rad(__args['h'])
                targets.append(pm.Actor(p0[0], p0[1], v, h, __k))
        elif _k == 'drones':
            for __k, __args in _dict[_k].items():
                p0, v, h = __args['p0'], __args['v'], np.deg2rad(__args['h'])
                drones.append(pm.Drone(p0, v, h, __k))
        elif _k == 'solutions':
            for __k, __args in _dict[_k].items():
                solutions.append([__k, __args['duration'], __args['seq']])
    # replace target names with references to actual targets
    _ta_by_name = {_t.name:_t for _t in targets}
    for _s in solutions: 
        if type(_s[2][0]) == list:
            _s[2] = [[_ta_by_name[_tn] for _tn in _seq] for _seq in _s[2]]
        else:
            _s[2] = [_ta_by_name[_tn] for _tn in _s[2]]
    print(f'  {len(drones)} drones, {len(targets)} targets, {len(solutions)} known solutions')
    #pdb.set_trace()
    for _s in solutions: _print_solution(_s)
    return drones, targets, solutions

def _fmt_actor(_a):
    return f'  {_a.name}:\n    p0: [{_a.x0},{_a.y0}]\n    v: {_a.v}\n    h: {np.rad2deg(_a.psi)}\n\n'
def _fmt_sol(_seq):
    return f'{[[_t.name for _t in _s] for _s in _seq]}'  if type(_seq[0]) == list else f'{[_t.name for _t in _seq]}'
def save_scenario(filename, drones, targets, solutions=[]):
    print(f'saving scenario to: {filename}')
    txt = 'targets:\n'
    for _t in targets: txt += _fmt_actor(_t)
    txt += 'drones:\n'
    for _d in drones: txt +=  _fmt_actor(_d)
    if len(solutions)>0:
        txt += '\nsolutions:\n'
        for _name, _dur, _seq in solutions:
            txt+=f'  {_name}:\n    duration: {_dur}\n    seq: {_fmt_sol(_seq)}\n'
    with open(filename, 'w') as f:
        f.write(txt)


def format_seq(_s): return '-'.join([f'{int(_t.name):02d}' for _t in _s]) 
def format_seqs(_ss): return '['+'] ['.join([format_seq(_s) for _s in _ss])+']'
    

class Scenario:
    def __init__(self, **kwargs):
        if 'filename' in kwargs: self.load(kwargs['filename'])
    
    def load(self, filename):
       self.drones, self.targets, self.solutions = load_scenario(filename)
       if len(self.drones)>0: self.drone = self.drones[0] # try to not break everything, REMOVE ME
       self.name = f'{len(self.targets)}_{os.path.splitext(os.path.basename(filename))[0]}'
       self.solutions_by_name = {_s[0]: _s for _s in self.solutions}
       #self.solutions_by_cost = np.array(self.solutions, dtype=object)[np.argsort([_s[1] for _s in self.solutions])]

    def save(self, filename):
        try:
            save_scenario(filename, self.drones, self.targets, self.solutions)
        except AttributeError:
            save_scenario(filename, [self.drone], self.targets, self.solutions)

    def add_solution(self, name, dur, seq):
        self.solutions.append([name, dur, seq])
        self.solutions_by_name[name] = self.solutions[-1]
        # FIXME update self.solutions_by_cost
        #pdb.set_trace()

    def nb_solutions(self): return len(self.solutions)
    
    def solution_by_name(self, name):
        #if name == '_best': return self.solutions_by_cost[0]
        #else:
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

    

def merge(ds_ts):
    for i in range(1,len(ds_ts)):
        for _t in ds_ts[i][1]:
            _t.name= str(int(_t.name)+int(ds_ts[i-1][1][-1].name))
    res = []
    for _d, _t in ds_ts: res += _t
    return ds_ts[0][0], res


def _normal(_m, _s): return {'kind':'normal', 'mean':_m, 'std':_s}
def _circle(_r=50., _c=(0,0)): return {'kind':'circle', 'r':_r, 'center':_c}
def _line(_l=50, _c=(0,0), _g=np.pi/2): return {'kind':'line', 'len':_l, 'center':_c, 'gamma':_g}
def _away(_m=0, _s=0): return {'kind':'away', 'mean':_m, 'std':_s}
def _toward(_m=0, _s=0): return {'kind':'toward', 'mean':_m, 'std':_s}
def _grid(_d=15, _nr=5, _c=(0,0)): return {'kind':'grid', 'd':_d, 'nr':_nr, 'center':_c}
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
              lambda: make_random_scenario(ntarg=9, dp0=(10,0), dv=15, tp=_circle(25), th=_away(), tv=_normal(5., 0.))],
        94:  ['scenario_9_4.yaml', # line
             lambda: make_random_scenario(ntarg=9, dp0=(30,5), dv=15,
                                           tp={'kind':'line', 'len':50}, th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)})],
        96: ['scenario_9_6.yaml', # circle headings normal
             lambda: make_random_scenario(ntarg=9, dp0=(25,25), dv=15,
                                          tp=_circle(75), th=_normal(0., 0.), tv=_normal(5., 0.))],
        98:  ['scenario_9_8.yaml', # grid 1
             lambda: make_random_scenario(ntarg=9, dp0=(-10,-10), dv=15,
                                           tp={'kind':'grid', 'nr':3, 'd':15},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        99:  ['scenario_9_9.yaml', # grid 2
              lambda: make_random_scenario(ntarg=9, dp0=(-10,-10), dv=15, tp={'kind':'grid', 'nr':3, 'd':15},
                                           th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],

        
        101: ['scenario_10_1.yaml', # 10 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=10)],
        103:  ['scenario_10_3.yaml', # circle away
               lambda: make_random_scenario(ntarg=10, dp0=(10,0), dv=15, tp=_circle(25), th=_away(), tv=_normal(5., 0.))],
        104: ['scenario_10_4.yaml', # line
              lambda: make_random_scenario(ntarg=10, dp0=(30,5), dv=15,
                                           tp={'kind':'line', 'len':50}, th=_normal(0., 0.), tv=_normal(5., 0.))],
        106: ['scenario_10_6.yaml', # circle headings normal
             lambda: make_random_scenario(ntarg=10, dp0=(25,25), dv=15, tp=_circle(75), th=_normal(0., 0.), tv=_normal(5., 0.))],           
        108: ['scenario_10_8.yaml', # grid 1
              lambda: make_random_scenario(ntarg=10, dp0=(-10,-10), dv=15,
                                            tp={'kind':'grid', 'nr':3, 'd':15},
                                            th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv=ScenarioFactory._def_tv)],
        109: ['scenario_10_9.yaml', # grid 2
              lambda: make_random_scenario(ntarg=10, dp0=(-10,-10), dv=15, tp={'kind':'grid', 'nr':3, 'd':15},
                                           th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],

        116: ['scenario_11_6.yaml', # circle headings normal
             lambda: make_random_scenario(ntarg=11, dp0=(25,25), dv=15, tp=_circle(75), th=_normal(0., 0.), tv=_normal(5., 0.))],           
        119: ['scenario_11_9.yaml', # grid 2
              lambda: make_random_scenario(ntarg=11, dp0=(-10,-10), dv=15, tp={'kind':'grid', 'nr':3, 'd':15},
                                           th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],
        129: ['scenario_12_9.yaml', # grid 2
              lambda: make_random_scenario(ntarg=12, dp0=(-10,-10), dv=15, tp={'kind':'grid', 'nr':3, 'd':15},
                                           th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],


        151: ['scenario_15_1.yaml', # 15 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=15)],
        152: ['scenario_15_2.yaml', # circle toward
              lambda: make_random_scenario(ntarg=15, dp0=(30,5), dv=15, tp=_circle(200.), th=_toward(np.deg2rad(45.)), tv=_normal(5., 0.))],
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
        159: ['scenario_15_9.yaml', # grid 2
             lambda: make_random_scenario(ntarg=15, dp0=(-10,-10), tp={'kind':'grid', 'nr':4, 'd':15}, th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],
        1510: ['scenario_15_10.yaml', # grid
             lambda: make_random_scenario(ntarg=15, dp0=(-10,-10), dv=15,
                                          tp={'kind':'grid', 'nr':5, 'd':15},
                                          th={'kind':'normal', 'mean':np.deg2rad(20.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],

        301: ['scenario_30_1.yaml', # 30 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=30, tp={'kind':'uniform', 'low':-100, 'high':100})],
        302: ['scenario_30_2.yaml', # circle toward
              lambda: make_random_scenario(ntarg=30, dp0=(30,5), dv=15, tp=_circle(200.), th=_toward(), tv=_normal(5., 0.))],
        3021: ['scenario_30_2_1.yaml', # circle toward skewed
               lambda: make_random_scenario(ntarg=30, tp=_circle(100, (0,0)), th=_toward(np.deg2rad(45.), np.deg2rad(0.)),  tv=_normal(5., 0.))],
        3022: ['scenario_30_2_2.yaml', # circle toward skewed + line
               lambda: merge((make_random_scenario(ntarg=20, tp=_circle(100, (0,0)), th=_toward(np.deg2rad(45.), np.deg2rad(0.)),  tv=_normal(5., 0.)),
                              make_random_scenario(ntarg=10, tp=_line(100), th=_normal(np.deg2rad(0.), np.deg2rad(0.)),  tv=_normal(5., 0.))))],
        303: ['scenario_30_3.yaml', # circle away
              lambda: make_random_scenario(ntarg=30, dp0=(10,0), tp=_circle(25), th=_away(), tv=_normal(5.,0))],
        3032: ['scenario_30_3_2.yaml', # 3 circles
               lambda: merge((make_random_scenario(ntarg=10, dp0=(10,0), dv=10, tp=_circle(50, (-100, -20)), th=_normal(np.deg2rad( 30), 0)),
                              make_random_scenario(ntarg=10,                    tp=_circle(50, ( 100, -20)), th=_normal(np.deg2rad(150), 0)),
                              make_random_scenario(ntarg=10,                    tp=_circle(50, (   0, 140)), th=_normal(np.deg2rad(-90), 0))))],
        3033: ['scenario_30_3_3.yaml', # 3 circles
               lambda: merge((make_random_scenario(ntarg=10, dp0=(10,0), dv=15),
                              make_random_scenario(ntarg=20, dp0=(10,0), dv=15, tp=_circle(75), th=_normal(np.deg2rad(20.), 0.))))],
        
        304: ['scenario_30_4.yaml', # line
              lambda: make_random_scenario(ntarg=30, dp0=(50,0), tp=_line(150, (-100,0)), th=_normal(np.deg2rad(0.), 0.), tv=_normal(5., 0.))],
        306: ['scenario_30_6.yaml', # circle headings normal
              lambda: make_random_scenario(ntarg=30, dp0=(25,25), dv=15, tp=_circle(75), th=_normal(0., 0.), tv=_normal(5., 0.))],

        3061: ['scenario_30_6_1.yaml', # circle headings normal - 100m to be the same as 60 targets
              lambda: make_random_scenario(ntarg=30, dp0=(25,25), dv=15, tp=_circle(100), th=_normal(0., 0.), tv=_normal(5., 0.))],
        3062: ['scenario_30_6_2.yaml', # circle headings normal with noise
               lambda: make_random_scenario(ntarg=30, dp0=(25,25), dv=15, tp=_circle(100), th=_normal(0., np.deg2rad(5.)), tv=_normal(5., 0.5))],
        3063: ['scenario_30_6_3.yaml', # 3 circles
               lambda: merge((make_random_scenario(ntarg=10, dp0=(0,0), dv=15, tp=_circle(75, (-100,  100)), th=_normal(np.deg2rad(0.), 0.)),
                              make_random_scenario(ntarg=10,                   tp=_circle(75, ( 100, 0)), th=_normal(np.deg2rad(-180.), 0.)),
                              make_random_scenario(ntarg=10,                   tp=_circle(75, (-100, -100)), th=_normal(np.deg2rad(0.), 0.))))],
        307: ['scenario_30_7.yaml', # circle headings normal, faster
             lambda: make_random_scenario(ntarg=30, dp0=(25,25), dv=15,
                                           tp={'kind':'circle', 'r':75},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':7.5, 'std':0.})],
        308: ['scenario_30_8.yaml', # grid 1
             lambda: make_random_scenario(ntarg=30, dp0=(-10,-10), dv=15,
                                           tp={'kind':'grid', 'nr':5, 'd':15},
                                           th={'kind':'normal', 'mean':np.deg2rad(0.), 'std':np.deg2rad(0.)}, tv={'kind':'normal', 'mean':5., 'std':0.})],
        309: ['scenario_30_9.yaml', # grid 2
             lambda: make_random_scenario(ntarg=30, dp0=(-10,-10), tp={'kind':'grid', 'nr':5, 'd':15}, th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],
        3091: ['scen_30/9_1.yaml', # grid 2 starting south west
             lambda: make_random_scenario(ntarg=30, dp0=(0, 0), tp= _grid(_d=15, _nr=5, _c=(-120,-80)), th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],
        3092: ['scen_30/9_2.yaml', # 2 grids colliding
             lambda: merge((make_random_scenario(ntarg=15, dp0=(0, 0), tp= _grid(_d=15, _nr=4, _c=(-50,-50)), th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.)),
                            make_random_scenario(ntarg=15, dp0=(0, 0), tp= _grid(_d=15, _nr=4, _c=(30,30)), th=_normal(np.deg2rad(-100.), 0.), tv=_normal(5., 0.))))],
        3093: ['scen_30/9_3.yaml', # 3 grids colliding
             lambda: merge((make_random_scenario(ntarg=10, dp0=(0, 0), tp= _grid(_d=15, _nr=3, _c=(-50,-54)), th=_normal(np.deg2rad(00.), 0.), tv=_normal(5., 0.)),
                            make_random_scenario(ntarg=10, dp0=(0, 0), tp= _grid(_d=15, _nr=3, _c=( 50,-46)), th=_normal(np.deg2rad(180.), 0.), tv=_normal(5., 0.)),
                            make_random_scenario(ntarg=10, dp0=(0, 0), tp= _grid(_d=15, _nr=3, _c=(  0, 30)), th=_normal(np.deg2rad(-90.), 0.), tv=_normal(5., 0.))))],
        601: ['scenario_60_1.yaml', # 60 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=60, tp={'kind':'uniform', 'low':-100, 'high':100})],
        602: ['scenario_60_2.yaml', # circle toward skewed
              lambda: make_random_scenario(ntarg=60, dp0=(30,5), dv=15,
                                           tp=_circle(100, (0,0)), th=_toward(np.deg2rad(45.), np.deg2rad(0.)),  tv=_normal(5., 0.))],
        6022: ['scenario_60_2_2.yaml', # circle toward skewed + line
               lambda: merge((make_random_scenario(ntarg=40, tp=_circle(100, (0,0)), th=_toward(np.deg2rad(45.), np.deg2rad(0.)),  tv=_normal(5., 0.)),
                              make_random_scenario(ntarg=20, tp=_line(100), th=_normal(np.deg2rad(0.), np.deg2rad(0.)),  tv=_normal(5., 0.))))],
        6023: ['scenario_60_2_3.yaml', # circle toward skewed + two lines
               lambda: merge((make_random_scenario(ntarg=40, tp=_circle(100, (0,0)), th=_toward(np.deg2rad(45.), np.deg2rad(0.)),  tv=_normal(5., 0.)),
                              make_random_scenario(ntarg=10, tp=_line(100, (0,-120),  0), th=_normal(np.deg2rad(90.), np.deg2rad(0.)),  tv=_normal(5., 0.)),
                              make_random_scenario(ntarg=10, tp=_line(100, (-120,0), np.pi/2), th=_normal(np.deg2rad(0.), np.deg2rad(0.)),  tv=_normal(5., 0.))))],
        603: ['scenario_60_3.yaml', # circle away
              lambda: make_random_scenario(ntarg=60, dp0=(0,0), tp=_circle(25), th=_away(), tv=_normal(5.,0))],
        604: ['scenario_60_4.yaml', # line
              lambda: make_random_scenario(ntarg=60, dp0=(50,0), tp=_line(150), th=_normal(np.deg2rad(0.), 0.), tv=_normal(5., 0.))],
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
        6063: ['scenario_60_6_3.yaml', # 3 circles
               lambda: merge((make_random_scenario(ntarg=20, dp0=(0,0), dv=15, tp=_circle(200, (0,  0)), th=_toward(np.deg2rad(20.), np.deg2rad(0.))),
                              make_random_scenario(ntarg=20,                   tp=_circle(75, ( 150, 50)), th=_normal(np.deg2rad(-180.), 0.)),
                              make_random_scenario(ntarg=20,                   tp=_circle(75, (-150, -50)), th=_normal(np.deg2rad(0.), 0.))))],

        609:  ['scenario_60_9.yaml', # grid 2
               lambda: make_random_scenario(ntarg=60, dp0=(-10,-10), tp={'kind':'grid', 'nr':8, 'd':15}, th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],
        6091:  ['scenario_60_9_1.yaml', # grid 2 with noise
               lambda: make_random_scenario(ntarg=60, dp0=(-10,-10), tp={'kind':'grid', 'nr':8, 'd':15}, th=_normal(np.deg2rad(20.), np.deg2rad(10.)), tv=_normal(5., 0.5))],
        6092:  ['scen_60/9_2.yaml', # grid 2 targets stopped
               lambda: make_random_scenario(ntarg=60, dp0=(-10,-10), tp={'kind':'grid', 'nr':8, 'd':15}, th=_normal(np.deg2rad(20.), np.deg2rad(0.)), tv=_normal(0., 0.))],

        
        1201: ['scenario_120_1.yaml', # 120 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=120, tp={'kind':'uniform', 'low':-250, 'high':250})],
        1202: ['scenario_120_2.yaml', # circle toward
               lambda: make_random_scenario(ntarg=120, dp0=(30,5), dv=15,
                                            tp=_circle(100, (0,0)), th=_toward(np.deg2rad(45.), np.deg2rad(0.)),  tv=_normal(5., 0.))],
        12022: ['scenario_120_2_2.yaml', # circle toward skewed + line
                lambda: merge((make_random_scenario(ntarg=80, tp=_circle(100, (0,0)), th=_toward(np.deg2rad(45.), np.deg2rad(0.)),  tv=_normal(5., 0.)),
                               make_random_scenario(ntarg=40, tp=_line(100), th=_normal(np.deg2rad(0.), np.deg2rad(0.)),  tv=_normal(5., 0.))))],
        1203: ['scenario_120_3.yaml', # circle away
              lambda: make_random_scenario(ntarg=120, dp0=(10,0), tp=_circle(25), th=_away(), tv=_normal(5.,0))],
        1204: ['scenario_120_4.yaml', # line
               lambda: make_random_scenario(ntarg=120, dp0=(50,0), tp=_line(150, (-100,0)), th=_normal(np.deg2rad(0.), 0.), tv=_normal(5., 0.))],
        1205: ['scenario_120_5.yaml', # 120 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=120, tp={'kind':'uniform', 'low':-250, 'high':250})],
        1206: ['scenario_120_6.yaml', # circle headings normal
               lambda: make_random_scenario(ntarg=120, tp=_circle(200), th=_normal(0., 0.), tv=_normal(5., 0.))],
        1209:  ['scenario_120_9.yaml', # grid 2
                lambda: make_random_scenario(ntarg=120, dp0=(-10,-10), tp={'kind':'grid', 'nr':11, 'd':15}, th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],
        
        2401: ['scenario_240_1.yaml', # 240 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=240, tp={'kind':'uniform', 'low':-250, 'high':250})],
        2402: ['scenario_240_2.yaml', # circle toward
               lambda: make_random_scenario(ntarg=240, tp=_circle(200, (0,0)), th=_toward(np.deg2rad(45.), np.deg2rad(0.)),  tv=_normal(5., 0.))],
        24022: ['scen_240/2_2.yaml', # circle toward skewed + line
                lambda: merge((make_random_scenario(ntarg=160, tp=_circle(100, (0,0)), th=_toward(np.deg2rad(45.), np.deg2rad(0.)),  tv=_normal(5., 0.)),
                               make_random_scenario(ntarg=80, tp=_line(100), th=_normal(np.deg2rad(0.), np.deg2rad(0.)),  tv=_normal(5., 0.))))],
        2403: ['scenario_240_3.yaml', # circle away
              lambda: make_random_scenario(ntarg=240, dp0=(10,0), tp=_circle(25), th=_away(), tv=_normal(5.,0))],
        2404: ['scenario_240_4.yaml', # line
               lambda: make_random_scenario(ntarg=240, dp0=(50,0), tp=_line(150, (-100,0)), th=_normal(np.deg2rad(0.), 0.), tv=_normal(5., 0.))],
        2406: ['scenario_240_6.yaml', # circle headings parallel
               lambda: make_random_scenario(ntarg=240, tp=_circle(300), th=_normal(0., 0.), tv=_normal(5., 0.))],
        2409:  ['scenario_240_9.yaml', # grid 2
                lambda: make_random_scenario(ntarg=240, dp0=(-10,-10), tp={'kind':'grid', 'nr':16, 'd':15}, th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],

        4801: ['scenario_480_1.yaml', # 480 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=480, tp={'kind':'uniform', 'low':-250, 'high':250})],
        48022: ['scen_480/2_2.yaml', # circle toward skewed + line
                lambda: merge((make_random_scenario(ntarg=320, tp=_circle(100, (0,0)), th=_toward(np.deg2rad(45.), np.deg2rad(0.)),  tv=_normal(5., 0.)),
                               make_random_scenario(ntarg=160, tp=_line(100), th=_normal(np.deg2rad(0.), np.deg2rad(0.)),  tv=_normal(5., 0.))))],
        4804: ['scenario_480_4.yaml', # line
               lambda: make_random_scenario(ntarg=480, dp0=(50,0), tp=_line(150, (-100,0)), th=_normal(np.deg2rad(0.), 0.), tv=_normal(5., 0.))],
        4806: ['scenario_480_6.yaml', # circle headings parallel
               lambda: make_random_scenario(ntarg=480, tp=_circle(300),  th=_normal(0., 0.), tv=_normal(5., 0.))],
        4809: ['scenario_480_9.yaml', # grid 2
                lambda: make_random_scenario(ntarg=480, dp0=(-10,-10), tp={'kind':'grid', 'nr':22, 'd':15}, th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],
        48091: ['scenario_48_9_1.yaml', # grid 2 with noise
               lambda: make_random_scenario(ntarg=480, dp0=(-10,-10), tp={'kind':'grid', 'nr':22, 'd':15}, th=_normal(np.deg2rad(20.), np.deg2rad(10.)), tv=_normal(5., 0.5))],

        9601: ['scenario_960_1.yaml', # 960 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=960, tp={'kind':'uniform', 'low':-250, 'high':250})],
        9604: ['scenario_960_4.yaml', # line
               lambda: make_random_scenario(ntarg=960, dp0=(50,0), tp=_line(150, (-100,0)), th=_normal(np.deg2rad(0.), 0.), tv=_normal(5., 0.))],
        9606: ['scenario_960_6.yaml', # circle headings parallel
               lambda: make_random_scenario(ntarg=960, tp=_circle(300),  th=_normal(0., 0.), tv=_normal(5., 0.))],
        9609: ['scenario_960_9.yaml', # grid 2
               lambda: make_random_scenario(ntarg=960, dp0=(-10,-10), tp={'kind':'grid', 'nr':31, 'd':15}, th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],
        19201: ['scenario_1920_1.yaml', # 1920 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=1920, tp={'kind':'uniform', 'low':-250, 'high':250})],
        19206: ['scenario_1920_6.yaml', # circle headings parallel
               lambda: make_random_scenario(ntarg=1920, tp=_circle(300), th=_normal(0., 0.), tv=_normal(5., 0.))],

        38401: ['scenario_3840_1.yaml', # 3840 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=3840, tp={'kind':'uniform', 'low':-250, 'high':250})],
        38406: ['scenario_3840_6.yaml', # circle headings parallel
               lambda: make_random_scenario(ntarg=3840, tp=_circle(300), th=_normal(0., 0.), tv=_normal(5., 0.))],

        78201: ['scenario_7820_1.yaml', # 7820 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=7820, tp={'kind':'uniform', 'low':-250, 'high':250})],
        78209: ['scen_7820/9.yaml', # grid 2
                lambda: make_random_scenario(ntarg=7820, dp0=(-10,-10), tp={'kind':'grid', 'nr':89, 'd':15}, th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],

        156401: ['scenario_15640_1.yaml', # 15640 targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=15640, tp={'kind':'uniform', 'low':-250, 'high':250})],
        156409: ['scen_15640/9.yaml', # grid 2
                lambda: make_random_scenario(ntarg=15640, dp0=(-10,-10), tp={'kind':'grid', 'nr':125, 'd':15}, th=_normal(np.deg2rad(20.), 0.), tv=_normal(5., 0.))],
        500001: ['scen_50k/1.yaml', # 50k targets, defaults: uniform law for pos and heading, normal law for speed
              lambda: make_random_scenario(ntarg=50000, tp={'kind':'uniform', 'low':-250, 'high':250})],
    }


    def filename(idx): return ScenarioFactory.scenarios[idx][0]
    
    def make(idx):
        _filename, _f = ScenarioFactory.scenarios[idx]
        drone, targets = _f()
        s = Scenario()
        s.drone, s.targets, s.solutions, s.solutions_by_name = drone, targets, [], {}
        #s.save(ScenarioFactory.filenames[idx])
        return s



