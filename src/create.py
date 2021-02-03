#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Create a scenario
  (see ScenarioFactory in proj_manen.utils)
'''

import sys, argparse, os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu

def main(idx, show=False):
    scen = pmu.ScenarioFactory.make(idx)
    scen.save(f'/tmp/{pmu.ScenarioFactory.filename(idx)}')
    if show:
        pmu.plot_scenario(scen)
        plt.show()
    
if __name__ == '__main__':
    idx = 0
    if len(sys.argv) > 1: idx = int(sys.argv[1])
    main(idx, show=True)
