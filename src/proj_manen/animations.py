#! /usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools

import pdb

import proj_manen as pm, proj_manen.utils as pmu

# make a side-by-side animation with several solutions of one scenario 
def animate_solutions(scen, names, tf=1., window_title=None):

    sols = [scen.solution_by_name(name) for name in names]
    seqs = [_seq for _1, _2, _seq in sols]
    drones = [pm.intercept_sequence_copy(scen.drone, _seq)[0] for _seq in seqs] 

    _n = len(names)
    fig = plt.figure(tight_layout=True, figsize=[5.12*_n, 5.12])
    if window_title is not None: fig.canvas.set_window_title(window_title)

    axes = fig.subplots(1,_n)#, sharex=True)
    if _n == 1: axes=[axes]
    return animate_multi(fig, axes, drones, seqs, names, tf=tf)

# make a side-by-side animation with several scenarios
def animate_scenarios(scens, sol_names, tf=1.):
    nr, nc = len(scens), 1
    fig = plt.figure(tight_layout=True, figsize=[5.12*nc, 5.12*nr])
    axes = fig.subplots(nr, nc)
    sol_name = sol_names[0]
    drones, seqs, names = [], [], []
    for scen in scens:
        seqs.append(scen.solution_by_name(sol_name)[2])
        drones.append(pm.intercept_sequence_copy(scen.drone, seqs[-1])[0])
        names.append(f'{scen.name} / {sol_name}')
    return animate_multi(fig, axes, drones, seqs, names, tf=tf)
    
# compute scene extends
# def compute_and_set_scale(axes, drones, targets):
#     for _drone, _targets, _ax in zip(drones, targets, axes):
#         ps = np.array(_drone.Xs) # drone positions
#         t0s = np.array([_t.p0 for _t in _targets]) # targets initial positions
#         ps = np.append(ps, t0s, axis=0)
#         #pdb.set_trace()
#         margin = 10.
#         bl, tr = np.min(ps,axis=0)-margin, np.max(ps,axis=0)+margin
#         #print(f'extends {bl} {tr}')
#         xlim, ylim = (bl[0], tr[0]), (bl[1], tr[1])
#         #print(f'xlim {xlim} ylim {ylim}')
#         dx, dy = tr-bl
#         if dx>dy:
#             _c = np.sum(ylim); ylim= ((_c-dx)/2, (_c+dx)/2)
#         else:
#             _c = np.sum(xlim); xlim= ((_c-dy)/2, (_c+dy)/2)
#         _ax.set_xlim(*xlim)
#         _ax.set_ylim(*ylim)
#         _ax.axis('equal')

def make_square_lims(tr, bl):
    xlim, ylim = (bl[0], tr[0]), (bl[1], tr[1])
    dx, dy = tr-bl
    if dx>dy:
        _c = np.sum(ylim); ylim= ((_c-dx)/2, (_c+dx)/2)
    else:
        _c = np.sum(xlim); xlim= ((_c-dy)/2, (_c+dy)/2)
    return xlim, ylim
    
        
def compute_and_set_scale(anims, mode='each'):
    if mode == 'each':
        for _a in anims:
            pos = np.append(np.array(_a.drone.Xs), np.array([_t.p0 for _t in _a.targets]), axis=0)
            margin = 10.
            bl, tr = np.min(pos,axis=0)-margin, np.max(pos,axis=0)+margin
            xlim, ylim =  make_square_lims(tr, bl)
            _a.ax.set_xlim(*xlim)
            _a.ax.set_ylim(*ylim)
            _a.ax.axis('equal')
    elif mode == 'same':
        pos = np.vstack([np.append(np.array(_a.drone.Xs), np.array([_t.p0 for _t in _a.targets]), axis=0) for _a in anims])
        margin = 10.
        bl, tr = np.min(pos,axis=0)-margin, np.max(pos,axis=0)+margin
        xlim, ylim =  make_square_lims(tr, bl)
        for _a in anims:
            _a.ax.set_xlim(*xlim)
            _a.ax.set_ylim(*ylim)
            _a.ax.axis('equal')

_COL_ON = 'wheat'
_COL_OFF = 'green'
_COL_DRONE = 'C3'
class MyAnimation:
    def __init__(self, ax, drone, targets, name):
        self.ax, self.drone, self.targets = ax, drone, targets
        self.status_fmt = f'Time: {{:04.1f}} / {drone.flight_duration():.2f} s   Targets: {{}} / {len(targets)}'
        bbox_props = dict(boxstyle='round', alpha=0.5)
        self.status_text = ax.text(0.025, 0.92, '', transform=ax.transAxes, bbox=bbox_props)
        self.circle_drone = plt.Circle((0, 0), 2., color=_COL_DRONE, fill=False, zorder=1)
        self.line_drone = ax.add_artist(self.circle_drone)
        self.circle_targets = []
        
    def init(self):
        self.finished = False
        self.status_text.get_bbox_patch().set_color(_COL_ON)
        return [self.status_text, self.line_drone]

    def update(self, i, t):
        idx_leg = self.drone._idx_leg(t)%len(self.drone.ts)
        if not self.finished:
            self.status_text.set_text(self.status_fmt.format(t, idx_leg))
            self.circle_drone.center = self.drone.get_pos(t)
            if t >= self.drone.flight_duration():
                self.status_text.get_bbox_patch().set_color(_COL_OFF)
                self.finished = True
        return [self.status_text, self.line_drone]

    
def animate_multi(fig, axes, drones, targets, names, t0=0., t1=None, dt=0.1, xlim=(-200, 200), ylim=(-200, 200), tf=1.):
    if t1 is None: t1 = np.max([_d.flight_duration() for _d in drones])
    anims = [MyAnimation(_ax, _d, _t, _n) for _ax, _d, _t, _n in zip(axes, drones, targets, names)]
    compute_and_set_scale(anims, mode='same')

    time = np.arange(t0, t1+dt, dt)
    _circle_targets = []
    for _ts in targets:
        _circle_targets.append([plt.Circle((_t.x0, _t.y0), 2., color='C1', fill=True, zorder=1)  for _t in _ts])
    _line_targets = []
    for _ax, _ct in zip(axes, _circle_targets):
        _line_targets.append([_ax.add_artist(_c) for _c in _ct])
    _all_l = [_l for _ls in _line_targets for _l in _ls ]
    _line_targets = np.array(_line_targets).flatten().tolist()

    #_all_c = list(itertools.chain.from_iterable(_circle_targets))
    _all_c = [_c for _cs in _circle_targets for _c in _cs ]
    #_all_c = np.array(_circle_targets).flatten().tolist()
    #pdb.set_trace()
    
    for _ax, _name in zip(axes, names):
        _ax.grid(); _ax.set_title(_name)#; _ax.axis('equal')#; _ax.set_xlim(*xlim); _ax.set_ylim(*ylim); 

    def _init():
        for _c in _all_c:# set all targets active
            _c.set_edgecolor('C1'); _c.set_facecolor('C1')
        foo=[]
        for _a in anims:
            foo+=_a.init()
        return foo  + _all_l#_line_targets#+ _line_drones

    def _animate(i):
        t = t0 + i * dt
        for _d, _ts, _cts in zip(drones, targets, _circle_targets):
            idx_leg = _d._idx_leg(t)%len(_d.ts)
            #if idx_leg < 0: idx_leg += len(_d.ts)
            if t <= _d.flight_duration() +0.1:
                for _k, (_t, _ct) in enumerate(zip(_ts, _cts)): # for all targets
                    if _k<idx_leg:
                        _ct.set_edgecolor('C2'); _ct.set_facecolor('C2') # turn inactive
                    else:
                       _ct.center = _t.get_pos(t)  # move target
        foo=[]
        for _a in anims:
            foo+=_a.update(i, t)
            
        return foo + _all_l#_line_targets# + _line_drones + + _status_texts

    anim = animation.FuncAnimation(fig, _animate, np.arange(1, len(time)),
                                   interval=dt*1e3/tf, blit=True, init_func=_init, repeat_delay=1000)
    
    return anim
    

def animate(fig, ax, drone=None, targets=None, t0=0., t1=10., dt=0.1, xlim=(-100, 100), ylim=(-100, 100), title=None):
    status_template = f'Time: {{:04.1f}} / {drone.flight_duration():.1f} s   Targets: {{}} / {len(targets)}'
    status_text = ax.text(0.025, 0.92, '', transform=ax.transAxes)
    time = np.arange(t0, t1, dt)
    _pd = drone.get_pos(t0)
    _circle_drone = plt.Circle((_pd[0], _pd[1]), 2., color='C0', fill=False, zorder=1)
    _line_drone = ax.add_artist(_circle_drone)
    _circle_targets = [plt.Circle((_t.x0, _t.y0), 2., color='C1', fill=True, zorder=1)  for _t in targets]
    _line_targets = [ax.add_artist(_c) for _c in _circle_targets]
        
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid()
    if title is not None: ax.set_title(title)

    def _init():
        status_text.set_text('N/A')
        for _c in _circle_targets:
            _c.set_edgecolor('C1'); _c.set_facecolor('C1')
        res = [status_text, _line_drone]+_circle_targets
        return res
    
    def _animate(i):
        t = t0 + i * dt; idx_leg = drone._idx_leg(t)
        status_text.set_text(status_template.format(t, idx_leg))
        if t <= drone.flight_duration():
            _circle_drone.center = drone.get_pos(t)
            for _k, (_c, _t) in enumerate(zip(_circle_targets, targets)):
                if _k>=idx_leg:
                    _c.center = _t.get_pos(t)#; _c.set_edgecolor('C1')
                else:
                    _c.set_edgecolor('C2'); _c.set_facecolor('C2')
        res = [status_text, _line_drone]+_circle_targets
        return res
    
    anim = animation.FuncAnimation(fig, _animate, np.arange(1, len(time)),
                                   interval=dt*1e3, blit=True, init_func=_init, repeat_delay=1000)
    
    return anim

def save_animation(anim, filename, dt):
    print('encoding animation video, please wait, it will take a while')
    anim.save(filename, writer='ffmpeg', fps=1./dt)
    #anim.save(filename, fps=1./dt, writer='imagemagick') # fails... 
    print('video encoded, saved to {}, Bye'.format(filename))

def main():
    #drone, targets = pm.load_scenario('./scenario_6.yaml')
    #drone, targets = pmu.make_random_scenario(10)
    drone, targets = pmu.make_conv_div_scenario(8)
    pm.intercept_sequence(drone, targets)#[::-1])
    print(f'duration: {drone.ts[-1]:.2f}s heading: {np.rad2deg(drone.psis[-1]):.1f} deg')
    anim = animate(plt.gcf(), plt.gca(), drone, targets,
                   t0=0., t1=drone.ts[-1], dt=0.1, xlim=(-100, 200), ylim=(-150, 150))
    #save_animation(anim, '/tmp/anim.mp4', dt=0.1)
    # ffmpeg -i /tmp/anim.mp4 -r 15 ../docs/images/animation_1.gif
    plt.show()

if __name__ == '__main__':
    main()
    
