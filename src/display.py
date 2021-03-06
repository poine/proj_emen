#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Display scenarios and solutions, as plot or animation
'''

import sys, argparse, os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.animations as pma, proj_manen.scenarios as pm_sc

def main(scen_filenames, sol_names,  anim=False, tf=1., save_anim=False, save_filename=None, scale=1.):
    #pdb.set_trace()
    scens = [pm_sc.Scenario(filename=_f) for _f in scen_filenames]
    if len(scens) > 1: # several scenarios
        anim = pma.animate_multi_scen_single_sol_md(scens, sol_names, tf)
    else:              # single scenario
        filename, scen = scen_filenames[0], scens[0]
        if scen.nb_solutions() == 0 or sol_names is None:  # no solution
            pmu.plot_scenario(scen, filename)
        else:                                              # solutions
            if anim:
                if len(sol_names) >= 2:
                    anim = pma.animate_solutions(scen, sol_names, tf, scen.name, scale*3.84)
                else:
                    anim = pma.animate_single_sol__md(scen, sol_names[0], tf, scen.name, scale*3.84)
            else :
                pmu.plot_solutions(scen, sol_names, filename)
    plt.show()
    if save_anim:
        #pma.save_animation(anim, '/tmp/anim2.mp4', dt=0.1/tf)  # ugly 
        #pma.save_animation(anim, '/tmp/anim.gif', dt=0.1/tf)  # unpractical
        pma.save_animation(anim, save_filename, dt=0.1/tf) # yes!!!

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display a scenario.')
    parser.add_argument("filename", nargs='*')
    parser.add_argument('-n', '--sol_name', help='name of solution to display')
    parser.add_argument('-a', '--anim', help='show animation', action="store_true")
    parser.add_argument('-f', '--time_factor', help='time factor', type=float, default=1.)
    parser.add_argument('-w', '--save', help='save animation', action="store_true")
    parser.add_argument('-o', '--out', help='name of file to save animation (.gif, .webp or .mp4)', default="/tmp/anim.webp")
    parser.add_argument('-s', '--scale', help='scale of the figure', type=float, default=1.)
    args = parser.parse_args()
    main(args.filename, args.sol_name.split(','), args.anim, args.time_factor, args.save, args.out, args.scale)
