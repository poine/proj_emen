#! /usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as animation

import pdb

import proj_manen as pm, proj_manen.utils as pmu


def animate_solutions(scen, names, window_title=None):

    sols = [scen.solution_by_name(name) for name in names]
    seqs = [_seq for _1, _2, _seq in sols]
    drones = [pm.intercept_sequence_copy(scen.drone, _seq)[0] for _seq in seqs] 

    _n = len(names)
    fig = plt.figure(tight_layout=True, figsize=[5.12*_n, 5.12])
    if window_title is not None: fig.canvas.set_window_title(window_title)

    axes = fig.subplots(1,_n)#, sharex=True)
    if _n == 1: axes=[axes]
    return animate_multi(fig, axes, drones, seqs, names)


# compute scene extends
def compute_and_set_scale(axes, drones, targets):
    for _drone, _targets, _ax in zip(drones, targets, axes):
        ps = np.array(_drone.Xs) # drone positions
        t0s = np.array([_t.p0 for _t in _targets]) # targets initial positions
        ps = np.append(ps, t0s, axis=0)
        #pdb.set_trace()
        margin = 10.
        bl, tr = np.min(ps,axis=0)-margin, np.max(ps,axis=0)+margin
        #print(f'extends {bl} {tr}')
        xlim, ylim = (bl[0], tr[0]), (bl[1], tr[1])
        #print(f'xlim {xlim} ylim {ylim}')
        dx, dy = tr-bl
        if dx>dy:
            _c = np.sum(ylim); ylim= ((_c-dx)/2, (_c+dx)/2)
        else:
            _c = np.sum(xlim); xlim= ((_c-dy)/2, (_c+dy)/2)
        _ax.set_xlim(*xlim)
        _ax.set_ylim(*ylim)
        _ax.axis('equal')


_COL_ON = 'wheat'
_COL_OFF = 'green'
class MyAnimation:
    def __init__(self, ax, drone, targets, name):
        self.drone, self.targets = drone, targets
        self.status_fmt = f'Time: {{:04.1f}} / {drone.flight_duration():.1f} s   Targets: {{}} / {len(targets)}'
        bbox_props = dict(boxstyle='round', alpha=0.5)
        self.status_text = ax.text(0.025, 0.92, '', transform=ax.transAxes, bbox=bbox_props)

    def init(self):
        self.finished = False
        self.status_text.get_bbox_patch().set_color(_COL_ON)
        return [self.status_text]

    def update(self, i, t):
        idx_leg = self.drone._idx_leg(t)%len(self.drone.ts)
        if t <= self.drone.flight_duration():
            self.status_text.set_text(self.status_fmt.format(t, idx_leg))
        else:
            if not self.finished:
                self.status_text.get_bbox_patch().set_color(_COL_OFF)
                self.finished = True
        return [self.status_text] # or []

    
def animate_multi(fig, axes, drones, targets, names, t0=0., t1=None, dt=0.1, xlim=(-200, 200), ylim=(-200, 200)):
    if t1 is None: t1 = np.max([_d.flight_duration() for _d in drones])
    compute_and_set_scale(axes, drones, targets)
    time = np.arange(t0, t1, dt)

    anims = [MyAnimation(_ax, _d, _t, _n) for _ax, _d, _t, _n in zip(axes, drones, targets, names)]
    
    #_status_fmts = [f'Time: {{:04.1f}} / {_d.flight_duration():.1f} s   Targets: {{}} / {len(_t)}' for _d, _t in zip(drones, targets)]
    #bbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #_status_texts = [_ax.text(0.025, 0.92, '', transform=_ax.transAxes, bbox=bbox_props) for _ax in axes]
    _circle_targets = []
    for _ts in targets:
        _circle_targets.append([plt.Circle((_t.x0, _t.y0), 2., color='C1', fill=True, zorder=1)  for _t in _ts])
    _line_targets = []
    for _ax, _ct in zip(axes, _circle_targets):
        _line_targets.append([_ax.add_artist(_c) for _c in _ct])
    _line_targets = np.array(_line_targets).flatten().tolist()
    _circle_drones = [plt.Circle((0, 0), 2., color='C3', fill=False, zorder=1) for _d in drones]
    _line_drones = [_ax.add_artist(_c) for _ax, _c in zip(axes, _circle_drones)]
    #pdb.set_trace()
        
    for _ax, _name in zip(axes, names):
        _ax.grid(); _ax.set_title(_name)#; _ax.axis('equal')#; _ax.set_xlim(*xlim); _ax.set_ylim(*ylim); 

    def _init():
        for _c in np.array(_circle_targets).flatten().tolist(): # set all targets active
            _c.set_edgecolor('C1'); _c.set_facecolor('C1')
        foo=[]
        for _a in anims:
            foo+=_a.init()
        return foo + _line_drones + _line_targets

    def _animate(i):
        t = t0 + i * dt
        for _d, _ts, _c, _cts in zip(drones, targets, _circle_drones, _circle_targets):
            idx_leg = _d._idx_leg(t)%len(_d.ts)
            #if idx_leg < 0: idx_leg += len(_d.ts)
            if t <= _d.flight_duration() +0.1:
                #_s.set_text(_sf.format(t, idx_leg))
                _c.center = _d.get_pos(t)
                for _k, (_t, _ct) in enumerate(zip(_ts, _cts)): # for all targets
                    if _k<idx_leg:
                        _ct.set_edgecolor('C2'); _ct.set_facecolor('C2') # turn inactive
                    else:
                       _ct.center = _t.get_pos(t)  # move target
            #else:
                #_s.get_bbox_patch().set_color('green')

        foo=[]
        for _a in anims:
            foo+=_a.update(i, t)
            
        return foo + _line_drones + _line_targets# + _status_texts

    anim = animation.FuncAnimation(fig, _animate, np.arange(1, len(time)),
                                   interval=dt*1e3, blit=True, init_func=_init, repeat_delay=1000)
    
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
    
