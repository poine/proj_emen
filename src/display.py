#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Display a scenario
'''

import sys, argparse, os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen_utils as pmu, animations as pma, test_3 as pm_t3

def main(filename, anim=False):
    scen = pmu.Scenario(filename=filename)
    sol_names = ['optimal', 'heuristic']
    if anim:
        anim = pm_t3.animate_solutions(scen, sol_names)
    else :
        pmu.plot_solutions(scen, sol_names, filename)
    plt.show()

if __name__ == '__main__':
    filename = pmu.ressource('data/scenario_3.yaml')
    if len(sys.argv) > 1: filename = sys.argv[1]
    main(filename)
