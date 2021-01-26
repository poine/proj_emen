#! /usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as animation

import pdb

import proj_manen as pm, proj_manen_utils as pmu

def animate_multi(fig, axes, drones, targets, names, t0=0., t1=None, dt=0.1, xlim=(-200, 200), ylim=(-200, 200)):
    if t1 is None: t1 = np.max([_d.flight_duration() for _d in drones])
    time = np.arange(t0, t1, dt)

    _status_fmts = [f'Time: {{:04.1f}} / {_d.flight_duration():.1f} s   Targets: {{}} / {len(_t)}' for _d, _t in zip(drones, targets)]
    _status_texts = [_ax.text(0.025, 0.92, '', transform=_ax.transAxes) for _ax in axes]
    _circle_drones = [plt.Circle((0, 0), 2., color='C3', fill=False, zorder=1) for _d in drones]
    _line_drones = [_ax.add_artist(_c) for _ax, _c in zip(axes, _circle_drones)]
    _circle_targets = []
    for _ts in targets:
        _circle_targets.append([plt.Circle((_t.x0, _t.y0), 2., color='C1', fill=True, zorder=1)  for _t in _ts])
    _line_targets = []
    for _ax, _ct in zip(axes, _circle_targets):
        _line_targets.append([_ax.add_artist(_c) for _c in _ct])
    _line_targets = np.array(_line_targets).flatten().tolist()
    #pdb.set_trace()
        
    for _ax, _name in zip(axes, names):
        _ax.set_xlim(*xlim); _ax.set_ylim(*ylim); _ax.grid(); _ax.set_title(_name); _ax.axis('equal')

    def _init():
        for _c in np.array(_circle_targets).flatten().tolist():
            _c.set_edgecolor('C1'); _c.set_facecolor('C1')
        return _line_drones + _line_targets

    def _animate(i):
        t = t0 + i * dt
        for _d, _ts, _s, _sf, _c, _cts in zip(drones, targets, _status_texts, _status_fmts, _circle_drones, _circle_targets):
            idx_leg = _d._idx_leg(t)
            _s.set_text(_sf.format(t, idx_leg))
            if t <= _d.flight_duration():
                _c.center = _d.get_pos(t)
                for _k, (_t, _ct) in enumerate(zip(_ts, _cts)):
                    if _k<idx_leg:
                        _ct.set_edgecolor('C2'); _ct.set_facecolor('C2')
                    else:
                       _ct.center = _t.get_pos(t)
                        
                
        return _line_drones + _line_targets + _status_texts

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
    
