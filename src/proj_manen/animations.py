#! /usr/bin/env python3
#-*- coding: utf-8 -*-

import time, numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools

import pdb

import proj_manen as pm, proj_manen.utils as pmu
import proj_manen.multi_drone as pm_md  # multi drone

# FIXME
def animate_single_sol__md(scen, sol_name, tf=1., window_title=None, _size=3.84):
    solution = scen.solution_by_name(sol_name)
    sequences = solution[2]
    if type(sequences[0]) != list: sequences=[sequences] # FIXME: make that homogeneous between single and multi cases
    #pdb.set_trace()
    drones, durs = pm_md.intercept_sequences_copy(scen.drones, sequences)
    fig = plt.figure(tight_layout=True, figsize=[_size, _size])
    return animate_fig(fig, [plt.gca()], [drones], [sequences], ['multi'], tf=tf)

# make a side-by-side animation with several scenarios (and one solution)
def animate_multi_scen_single_sol_md(scens, sol_names, tf=1., _size=3.84):
    nr, nc = 1, len(scens)
    fig = plt.figure(tight_layout=True, figsize=[_size*nc, _size*nr])
    axes = fig.subplots(nr, nc)
    sol_name = sol_names[0]
    drones, seqs, names = [], [], []
    for scen in scens:
        seqs.append(scen.solution_by_name(sol_name)[2])
        if type(seqs[-1][0]) != list: seqs[-1]=[seqs[-1]] # FIXME: make that homogeneous between single and multi cases
        drones.append(pm_md.intercept_sequences_copy(scen.drones, seqs[-1])[0])
        names.append(f'{scen.name} / {sol_name}')
    return animate_fig(fig, axes, drones, seqs, names, tf=tf)

# make a side-by-side animation with several solutions of one scenario 
def animate_solutions(scen, names, tf=1., window_title=None, _size=3.84):

    sols = [scen.solution_by_name(name) for name in names]
    seqs = [_seq for _1, _2, _seq in sols]
    drones = [pm.intercept_sequence_copy(scen.drone, _seq)[0] for _seq in seqs] 
    _n = len(names)
    fig = plt.figure(tight_layout=True, figsize=[_size*_n, _size])
    if window_title is not None: fig.canvas.set_window_title(window_title)

    axes = fig.subplots(1,_n)#, sharex=True)
    if _n == 1: axes=[axes]
    titles = [f'{window_title}/{_n}' for _n in names]
    return animate_fig(fig, axes, drones, seqs, titles, tf=tf)

    


# def get_ax_size(fig, ax):
#     bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     width, height = bbox.width, bbox.height
#     width *= fig.dpi
#     height *= fig.dpi
#     return width, height

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
            bl, tr =_a.get_extends()
            xlim, ylim =  make_square_lims(tr, bl)
            print(bl, tr, xlim, ylim)
            _a.ax.set_xlim(*xlim)
            _a.ax.set_ylim(*ylim)
            _a.ax.axis('equal')
    elif mode == 'same':
        extds = np.vstack([_a.get_extends() for _a in anims])
        bl, tr = np.min(extds,axis=0), np.max(extds,axis=0)
        xlim, ylim =  make_square_lims(tr, bl)
        for _a in anims:
            _a.ax.set_xlim(*xlim)
            _a.ax.set_ylim(*ylim)
            _a.ax.axis('equal')

_COL_ON = 'wheat'
_COL_OFF = 'green'
_COL_DRONE = 'C3'
_COL_TOFF = 'C2'
_COL_TON = 'C1'
class MyAnimation:
    def __init__(self, ax, drones, targets, name):
        self.ax, self.drones, self.targets = ax, drones, targets
        self.flight_durations = [_d.flight_duration() for _d in drones]
        self.flight_duration = np.max(self.flight_durations)
        self.status_fmt = f'Time: {{:04.1f}} / {self.flight_duration:.2f} s   Targets: {{}} / {len(targets)}'
        bbox_props = dict(boxstyle='round', alpha=0.5)
        self.status_text = ax.text(0.025, 0.92, '', transform=ax.transAxes, bbox=bbox_props)
        self.circle_drones = [plt.Circle((0, 0), 2., color=_COL_DRONE, fill=False, zorder=1) for _d in drones]
        self.line_drones = [ax.add_artist(_c) for _c in self.circle_drones]
        self.circle_targets = [[plt.Circle((_t.x0, _t.y0), 2., color=_COL_TOFF, fill=True, zorder=1)  for _t in _s] for _s in targets]
        self._circle_targets = [_c for _cs in self.circle_targets for _c in _cs]
        self.line_targets = [ax.add_artist(_c) for _c in self._circle_targets]

    def get_extends(self, margin=10.):
        drone_pos = np.vstack([np.array(_d.Xs) for _d in self.drones])
        #pdb.set_trace()
        #t_pos = np.vstack(np.array([_t.p0 for _t in _a.targets])
        pos = drone_pos
        bl, tr = np.min(pos,axis=0)-margin, np.max(pos,axis=0)+margin
        return bl, tr
        
    def init(self):
        self.finished = False
        self.status_text.get_bbox_patch().set_color(_COL_ON)
        for _c in self._circle_targets: # set all targets active
            _c.set_edgecolor(_COL_TON); _c.set_facecolor(_COL_TON)
        return [self.status_text] + self.line_drones + self.line_targets

    def update(self, i, t):
        idx_leg = [_d._idx_leg(t)%len(_d.ts) for _d in self.drones]
        if not self.finished:
            self.status_text.set_text(self.status_fmt.format(t, idx_leg))
            for _id, (_d, _cd, _ts, _cts) in enumerate(zip(self.drones, self.circle_drones, self.targets, self.circle_targets)):
                if t<= self.flight_durations[_id]:
                    _cd.center = _d.get_pos(t)
                idx_leg = _d._idx_leg(t)%len(_d.ts)
                for _i, (_c, _t) in enumerate(zip(_cts, _ts)):
                    if _i >= idx_leg:
                        _c.center = _t.get_pos(t)
                    else:
                        _c.set_edgecolor(_COL_TOFF); _c.set_facecolor(_COL_TOFF) # turn inactive
            if t >= self.flight_duration:
                self.status_text.get_bbox_patch().set_color(_COL_OFF)
                self.finished = True
        return [self.status_text] + self.line_drones + self.line_targets

# Handles all animations (in subplots).
# This is needed as matplotlib animation supports only one FuncAnimation    
def animate_fig(fig, axes, drones, targets, names, t0=0., t1=None, dt=0.1, xlim=(-200, 200), ylim=(-200, 200), tf=1.):
    if t1 is None: t1 = np.max([_d.flight_duration() for __d in drones for _d in __d])
    anims = [MyAnimation(_ax, _d, _t, _n) for _ax, _d, _t, _n in zip(axes, drones, targets, names)]
    compute_and_set_scale(anims, mode='same')
    time = np.arange(t0, t1+dt, dt)
    for _ax, _name in zip(axes, names):
        _ax.grid(); _ax.set_title(_name)

    def _init():
        res=[]
        for _a in anims: res += _a.init()
        return res

    def _animate(i):
        t = t0 + i * dt
        res=[]
        for _a in anims: res += _a.update(i, t)
        return res

    anim = animation.FuncAnimation(fig, _animate, np.arange(1, len(time)),
                                   interval=dt*1e3/tf, blit=True, init_func=_init, repeat_delay=1000)
    
    return anim
    

def save_animation(anim, filename, dt):
    print('encoding animation video, please wait, it will take a while')
    _start = time.time()
    #
    if filename.endswith('.mp4'):
        anim.save(filename, writer='ffmpeg', fps=1./dt)
    #    anim.save(filename, fps=1./dt, writer='imagemagick') # fails...
    elif filename.endswith('.gif') or filename.endswith('.webp'):
        fps = 1./dt; print(f'dt {dt} fps {fps}')
        anim.save(filename, writer=animation.PillowWriter(fps=1./dt)) # gif?
    _end = time.time()
    print(f'video encoded, saved to {filename}, Bye (took {_end-_start:.1f} s)')


    
def main():
    drone, targets = pm.load_scenario('./scenario_6.yaml')
    pm.intercept_sequence(drone, targets)#[::-1])
    print(f'duration: {drone.ts[-1]:.2f}s heading: {np.rad2deg(drone.psis[-1]):.1f} deg')
    anim = animate(plt.gcf(), plt.gca(), drone, targets,
                   t0=0., t1=drone.ts[-1], dt=0.1, xlim=(-100, 200), ylim=(-150, 150))
    #save_animation(anim, '/tmp/anim.mp4', dt=0.1)
    # ffmpeg -i /tmp/anim.mp4 -r 15 ../docs/images/animation_1.gif
    plt.show()

if __name__ == '__main__':
    main()
    
