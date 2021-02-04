#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Display a scenario
'''

import sys, argparse, os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.animations as pma

def main(filename, sol_names,  anim=False):
    #pdb.set_trace()
    scen = pmu.Scenario(filename=filename)
    if scen.nb_solutions() == 0 or sol_names is None:
        pmu.plot_scenario(scen, filename)
    else:
        sol_names = sol_names.split(',')
        if anim:
            anim = pma.animate_solutions(scen, sol_names, filename)
        else :
            pmu.plot_solutions(scen, sol_names, filename)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display a scenario.')
    parser.add_argument("filename")
    parser.add_argument('-n', '--sol_name', help='name of solution to display')
    parser.add_argument('-a', '--anim', help='show animation', action="store_true")
    args = parser.parse_args()
    main(args.filename, args.sol_name, args.anim)
