#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Display scenarios and solutions, as plot or animation
'''

import sys, argparse, os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.animations as pma

def main(filenames, sol_names,  anim=False, tf=1., save_anim=False, save_filename=None):
    #pdb.set_trace()
    if len(filenames) > 1:
        scens = [pmu.Scenario(filename=_f) for _f in filenames]
        sol_names = sol_names.split(',')
        anim = pma.animate_scenarios(scens, sol_names, tf)

    else:
        filename = filenames[0]
        scen = pmu.Scenario(filename=filename)
        if scen.nb_solutions() == 0 or sol_names is None:
            pmu.plot_scenario(scen, filename)
        else:
            sol_names = sol_names.split(',')
            if anim:
                anim = pma.animate_solutions(scen, sol_names, tf, scen.name)
            else :
                pmu.plot_solutions(scen, sol_names, filename)
    plt.show()
    if save_anim:
        #pma.save_animation(anim, '/tmp/anim2.mp4', dt=0.1/tf)  # ugly 
        #pma.save_animation(anim, '/tmp/anim.gif', dt=0.1/tf)  # unpractical
        pma.save_animation(anim, save_filename, dt=0.1/tf) # yes!!!

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display a scenario.')
    parser.add_argument("filename", nargs='*')#, action='append')
    parser.add_argument('-n', '--sol_name', help='name of solution to display')
    parser.add_argument('-a', '--anim', help='show animation', action="store_true")
    parser.add_argument('-f', '--time_factor', help='time factor', type=float, default=1.)
    parser.add_argument('-s', '--save', help='save animation', action="store_true")
    parser.add_argument('-o', '--out', help='name of file to save animation (.gif, .webp or .mp4)', default="/tmp/anim.webp")
    args = parser.parse_args()
    main(args.filename, args.sol_name, args.anim, args.time_factor, args.save, args.out)
