#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Make plots for documentation
'''
import os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.animations as pma



def plot_scenarios(idxs, sol_name):
    _fs = [pmu.ScenarioFactory.filename(idx) for idx in idxs]
    filenames = [pmu.ressource('data/'+_f) for _f in _fs]
    scens = [pmu.Scenario(filename=_f) for _f in filenames]
    pmu.plot_scenarios(scens, _fs, sol_name)
    plt.show()

def main():
    #plot_scenarios([71, 72, 73, 74, 78, 79], ['optimal'])
    #plot_scenarios([91, 92, 93, 94, 98, 99], 'optimal')
    #plot_scenarios([91, 93, 94, 96, 99], 'optimal')
    #plot_scenarios([101, 103, 104, 106, 109], 'optimal')
    #plot_scenarios([151, 152, 153], ['sa5k'])
    #plot_scenarios([301, 303, 304, 306, 309], 'sa5k')
    #plot_scenarios([306, 3061], 'sa5k')
    plot_scenarios([601, 603, 604, 606, 609], 'sa100k')
    #plot_scenarios([6022, 6023], 'sa100k6')
    #plot_scenarios([1202, 1206], 'best')
    
if __name__ == '__main__':
    main()
