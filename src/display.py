#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Display a scenario
'''

import sys, argparse, os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.animations as pma, test_3 as pm_t3

def main(filename, sol_names,  anim=False):
    #pdb.set_trace()
    scen = pmu.Scenario(filename=filename)
    if scen.nb_solutions() == 0 or sol_names is None:
        pmu.plot_scenario(scen, filename)
    else:
        sol_names = sol_names.split(',')#['optimal', 'heuristic']
        if anim:
            anim = pm_t3.animate_solutions(scen, sol_names)
        else :
            pmu.plot_solutions(scen, sol_names, filename)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display a scenario.')
    parser.add_argument("filename")
    parser.add_argument('-n', '--sol_name', help='name of solution to display')
    args = parser.parse_args()
    main(args.filename, args.sol_name)
