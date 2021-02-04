#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Make plots for documentation
'''
import os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.animations as pma



def plot_scenarios(idxs, sol_names):
    _fs = [pmu.ScenarioFactory.filename(idx) for idx in idxs]
    filenames = [pmu.ressource('data/'+_f) for _f in _fs]
    scens = [pmu.Scenario(filename=_f) for _f in filenames]
    pmu.plot_scenarios(scens, _fs, sol_names)
    plt.show()

def main():
    #plot_scenarios([71, 72, 73, 74, 78, 79], ['optimal'])
    plot_scenarios([91, 92, 93, 94, 98, 99], ['optimal'])
    #plot_scenarios([151, 152, 153], ['sa5k'])
    #plot_scenarios([301, 302, 303, 304, 308, 309], ['sa5k'])
    #plot_scenarios([606], ['sa5k'])
    
if __name__ == '__main__':
    main()