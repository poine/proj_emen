#! /usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as animation

import proj_manen as pm, proj_manen_utils as pmu

def animate(fig, ax, drone=None, targets=None, t0=0., t1=10., dt=0.1, xlim=(-100, 100), ylim=(-100, 100), title=None):
    status_template = f'Time: {{:04.1f}} / {drone.ts[-1]:.1f} s   Targets: {{}} / {len(targets)}'
    status_text = ax.text(0.025, 0.92, '', transform=ax.transAxes)
    time = np.arange(t0, t1, dt)
    _pd = drone.get_pos(t0)
    _circle_drone = plt.Circle((_pd[0], _pd[1]), 2., color='r', fill=False, zorder=1)
    _line_drone = ax.add_artist(_circle_drone)
    _circle_targets = [plt.Circle((_t.x0, _t.y0), 1.5, fill=False, zorder=1)  for _t in targets]
    _line_targets = [ax.add_artist(_c) for _c in _circle_targets]
        
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid()
    if title is not None: ax.set_title(title)

    def _init():
        status_text.set_text('N/A')
        res = [status_text, _line_drone]+_circle_targets
        return res
    
    def _animate(i):
        t = t0 + i * dt
        idx_leg = drone._idx_leg(t)
        status_text.set_text(status_template.format(t, idx_leg))
        if t <= drone.flight_duration():
            _circle_drone.center = drone.get_pos(t)
            for _k, (_c, _t) in enumerate(zip(_circle_targets, targets)):
                if _k>=idx_leg:
                    _c.center = _t.get_pos(t); _c.set_edgecolor('k')
                else:
                    _c.set_edgecolor('g')
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
    
