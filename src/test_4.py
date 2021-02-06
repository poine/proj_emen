#! /usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  Unit test for simulated annealing
'''
import os, time, copy, numpy as np, matplotlib.pyplot as plt
import pdb

import proj_manen as pm, proj_manen.utils as pmu, proj_manen.animations as pma, test_3 as pm_t3, proj_manen.simulated_annealing as pm_sa

# def _names_of_seq(_s): return [_t.name.split('_')[-1] for _t in _s]
# def _names_of_seq2(_s): foo = [int(_t.name.split('_')[-1]) for _t in _s]; return '-'.join([f'{_f:02d}' for _f in foo])
# def _print_sol(_d, _s): print(f'{_d: 8.2f}: {_names_of_seq2(_s)}')
# def _mutate(_seq):  # swaping to random stages
#     _seq2 = _seq.copy()
#     i1 = np.random.randint(0, high=len(_seq))
#     i2 = np.random.randint(0, high=len(_seq)-1)
#     while i2==i1: i2 = np.random.randint(0, high=len(_seq)-1)
#     _foo = _seq2.pop(i1); _seq2.insert(i2, _foo)
#     return _seq2

# def _mutate2(_seq):  # swaping to adjacent stages
#     _seq2 = _seq.copy()
#     i1 = np.random.randint(0, high=len(_seq))
#     i2 = i1-1 if i1>0 else i1+1
#     _foo = _seq2.pop(i1); _seq2.insert(i2, _foo)
#     return _seq2

# def search_locally(drone, targets, start_dur, start_seq, ntest, debug=False, Tf=None):
#     best_drone, cur_drone = drone, drone
#     best_seq = cur_seq = start_seq
#     best_dur = cur_dur = start_dur
#     if Tf is None: Tf = lambda i: ntest/(i+1)
#     if debug: all_durs, kept_durs, Paccept = np.zeros(ntest) , np.zeros(ntest), np.zeros(ntest) 
#     for i in range(ntest):
#         _s2 = _mutate(cur_seq)
#         _d2, _dur = pm.intercept_sequence_copy(drone, _s2)
#         if debug: all_durs[i] = _dur
#         if _dur < best_dur:
#             best_dur, best_drone, best_seq = _dur, _d2, _s2
#             #_print_sol(best_dur, best_seq)
#         T = Tf(i)
#         acc_prob = np.exp(-(_dur-cur_dur)/T) if _dur > cur_dur else 0. # warning 1
#         if debug: Paccept[i] = acc_prob
#         _r = np.random.uniform(low=0, high=1.)
#         if _dur < cur_dur or _r <= acc_prob:
#             cur_dur, cur_drone, cur_seq = _dur, _d2, _s2
#             _print_sol(cur_dur, cur_seq)
#         if debug: kept_durs[i] = cur_dur

#     if debug: return best_drone, best_seq, all_durs, kept_durs, Paccept
#     else: return best_drone, best_seq

def search_heuristic_closest_refined(scen, ntest, debug=False, Tf=None, start_seq_name='heuristic'):
    try: # compute closest_target heuristic if unknown
        name, heur_dur, heur_seq = scen.solution_by_name(start_seq_name)
    except KeyError:    
        heur_drone, heur_seq = pm_t3.search_heuristic_closest(scen.drone, scen.targets)
        heur_dur = heur_drone.flight_duration()
    print('Starting point (heuristic closest target)');pm_sa._print_sol(0, heur_dur, heur_dur, heur_seq)
    print(f'Local search ({ntest} iterations)')
    if debug:
        #best_drone, best_seq, all_durs, kept_durs, Paccept = search_locally(scen.drone, scen.targets, heur_dur, heur_seq, ntest, debug, Tf)
        best_drone, best_seq, all_durs, kept_durs, Paccept = pm_sa.search(scen.drone, scen.targets, start_dur=None, start_seq=None, ntest=ntest, debug=True, Tf=Tf, display=True)
        return best_drone, best_seq, all_durs, kept_durs, Paccept
    else:
        #best_drone, best_seq = search_locally(scen.drone, scen.targets, heur_dur, heur_seq, ntest)
        best_drone, best_seq = pm_sa.search(scen.drone, scen.targets, start_dur=None, start_seq=None, ntest=ntest, debug=False, Tf=Tf, display=False)
        return best_drone, best_seq





def test1(filename = 'scenario_30_1.yaml'):
    scen = pmu.Scenario(filename=filename)
    try:
        scen.solution_by_name('heuristic')
    except KeyError:
        test_drone, test_seq = pm_t3.search_heuristic_closest(scen.drone, scen.targets)
        scen.add_solution('heuristic', test_drone.flight_duration(), test_seq)
    
    best_drone, best_seq = search_heuristic_closest_refined(scen, ntest=10000)
    scen.add_solution('heuristic_refined', best_drone.flight_duration(), best_seq)
    scen.save(f'/tmp/{filename}')

def test2(filename = 'scenario_30_1.yaml'):
    scen = pmu.Scenario(filename=filename)
    try:
        _n, _d, best_seq = scen.solution_by_name('heuristic_refined')
        best_drone, _ = pm.intercept_sequence_copy(scen.drone, best_seq)
    except KeyError: 
        best_drone, best_seq = search_heuristic_closest_refined(scen, ntest=15000)
        scen.add_solution('heuristic_refined', best_drone.flight_duration(), best_seq)
        scen.save(f'/tmp/{filename}')
        
    names = ['heuristic', 'heuristic_refined', 'best']#, 'optimal']
    _n = len(names)
    if 0:
        fig = plt.figure(tight_layout=True, figsize=[5.12*_n, 5.12]);fig.canvas.set_window_title(filename)
        axes = fig.subplots(1,_n, sharex=True)
        pmu.plot_2d(fig, axes, scen, names)
    else:
        anim = pm_t3.animate_solutions(scen, ['heuristic', 'heuristic_refined'])
        

    
def test3(idx=17, show_search=True):
    if 1: scen = pmu.ScenarioFactory.make(idx)
    else: scen = pmu.Scenario(filename=pmu.ScenarioFactory.filenames[idx])

    heur_drone, heur_seq = pm_t3.search_heuristic_closest(scen.drone, scen.targets)
    scen.add_solution('heuristic', heur_drone.flight_duration(), heur_seq)

    if 0:#show_search:
        best_drone, best_seq = search_heuristic_closest_refined(scen, ntest=5000)
        scen.add_solution('heuristic_refined', best_drone.flight_duration(), best_seq)
        scen.save(f'/tmp/{pmu.ScenarioFactory.filenames[idx]}')
    if 0:
        anim = pm_t3.animate_solutions(scen, ['heuristic', 'best'])
        return anim
    if 1:
        pmu.plot_solutions(scen, ['heuristic'], pmu.ScenarioFactory.filenames[idx])
    #pmu.plot_solutions(scen, ['best', 'best1', 'best2'], pmu.ScenarioFactory.filenames[idx])
    #fig = plt.gcf()
    #if not show_search:
    #    pmu.plot_2d(fig, [plt.gca()], scen, ['heuristic'])
    #else:
    #    axes = fig.subplots(1,2, sharex=True)
    #    pmu.plot_2d(fig, axes, scen, ['heuristic', 'heuristic_refined'])

def play_anim(filename, sol_name):
    scen = pmu.Scenario(filename=filename)
    #pmu.plot_solutions(scen, [sol_name], filename)
    anim = pm_t3.animate_solutions(scen, [sol_name, sol_name])
    
def plot_search_chronograms(filename, epoch=100000):
    scen = pmu.Scenario(filename=filename)
    #pmu.plot_solutions(scen, ['best'], filename)
    #def aff(a0, a1, n, i): return a0 + (a1-a0)*i/ntest
    #def exp(e0, n, i): return e0*np.exp(-i/n)
    #def f1(a0, i0, a1, i1, i): return a0 if i<=i0 else a0+(i-i0)/(i1-i0)*(a1-a0) if i <= i1 else a1
    #Ts = [lambda i: 1e-16, lambda i: 10, lambda i: 100, lambda i: aff(100, 10, ntest, i)]
    Ts = [
        #lambda i: exp(100, ntest/3., i),
        #lambda i: exp(100, ntest/4., i),
        #lambda i: exp(100, ntest/6., i),
        #lambda i: exp(50, ntest/5., i),
        #lambda i: exp(50, epoch/6., i),
        #lambda i: exp(50, epoch/7., i),
        #lambda i: aff(100, 0.0005, epoch, i),
        #lambda i: f1(50, epoch/5, 1e-2, 8*epochst/10, i),
        #lambda i: pm_sa._f1(25, epoch/5, 1e-2, 8*epoch/10, i),
        lambda i: pm_sa._f1(5, epoch/5, 1e-2, 8*epoch/10, i),
        lambda i: pm_sa._f1(2, epoch/10, 1e-2, 9*epoch/10, i),
        #lambda i: pm_sa._f1(1, epoch/10, 1e-2, 9*epoch/10, i),
        #lambda i: f1(10, epoch/5, 1e-2, 8*epoch/10, i),
        #lambda i: aff(150, 0.0005, epoch, i),
        #lambda i: 1e-16,
    ]
    Ts = [lambda i: pm_sa._f1(2, epoch/10, 1e-2, 9*epoch/10, i)]*5

    ax1, ax2, ax3 = plt.gcf().subplots(3,1, sharex=True)
    for k, Tf in enumerate(Ts):
        best_drone, best_seq, all_durs, kept_durs, Paccept = pm_sa.search(scen.drone, scen.targets, ntest=epoch, debug=True, Tf=Tf, display=2)
        scen.add_solution('sa', best_drone.flight_duration(), best_seq)
        scen.save(f'/tmp/{os.path.basename(filename)}_{best_drone.flight_duration():.2f}')
        #plt.plot(all_durs)
        ax1.plot(kept_durs, label=f'{k}')
        ax2.plot(Paccept, label=f'{k}')
        ax3.plot([Ts[k](i) for i in range(epoch)], label=f'{k}')
        
        
    pmu.decorate(ax1, 'Cost', legend=True)#, ylim=(0, 1000))
    pmu.decorate(ax2, 'Paccept', legend=True)
    pmu.decorate(ax3, 'Temperature', legend=True)


# def create_or_load_search_set(scen_filename, nb_searches, epochs, cache_filename, force_run):
#     if force_run:
#         scen = pmu.Scenario(filename=scen_filename)
#         cost_by_ep, seq_by_ep = [],[]
#         for _ep in epochs:
#             print(f'-{_ep} epochs')
#             _drones, _seqs = [],[]
#             for i in range(nb_searches):
#                 _drone, _seq = pm_sa.search(scen.drone, scen.targets, ntest=_ep, display=1)
#                 _drones.append(_drone); _seqs.append(_seq)
#                 print(f' run {i: 3d}: {_drone.flight_duration():.2f}s')
#             cost_by_run = [_d.flight_duration() for _d in _drones]
#             cost_by_ep.append(cost_by_run)
#             seq_by_ep.append([pmu.format_seq(_s) for _s in _seqs])
#         np.savez(cache_filename, cost_by_ep=cost_by_ep, seq_by_ep=seq_by_ep, epochs=epochs)
#     else:
#         data = np.load(cache_filename+'.npz')
#         cost_by_ep, seq_by_ep, epochs = data['cost_by_ep'], data['seq_by_ep'], data['epochs'] 

#     return cost_by_ep, seq_by_ep, epochs

# def update_search_set(cache_filename1, cache_filename2, out_filename=None):
#     _cbe1, _sbe1, _e1 = create_or_load_search_set(None, None, None, cache_filename1, False)
#     _cbe2, _sbe2, _e2 = create_or_load_search_set(None, None, None, cache_filename2, False)
#     epochs = np.append(_e1, _e2)
#     cost_by_ep = np.append(_cbe1, _cbe2, axis=0)
#     seq_by_ep = np.append(_sbe1, _sbe2, axis=0)
#     if out_filename is not None:
#         np.savez(out_filename, cost_by_ep=cost_by_ep, seq_by_ep=seq_by_ep, epochs=epochs)
#     return cost_by_ep, seq_by_ep, epochs

# def plot_search_set(cost_by_ep, seq_by_ep, epochs, window_title):
#     for e, c in zip(epochs, cost_by_ep):
#         plt.hist(c, label=f'{e}', alpha=0.6)
#     pmu.decorate(plt.gca(), xlab='time in s', legend=True)
        
# def analyze_search_set(cost_by_ep, seq_by_ep, epochs, scen_filename=None, add_best=False, add_good=False, overwrite=False):
#     #print(f'{cost_by_ep}\n{seq_by_ep}\n{epochs}')
#     best_idx = np.argmin(cost_by_ep)
#     best_dur, best_seqn = cost_by_ep.flatten()[best_idx], seq_by_ep.flatten()[best_idx]
#     print(f'min cost {best_dur:.3f} {best_seqn}')
#     good_range = 1.1
#     good_idx = cost_by_ep <  best_dur*good_range
#     good_costs, good_seqns = cost_by_ep[good_idx], seq_by_ep[good_idx]
#     #pdb.set_trace()
#     print(f'found {len(good_costs)} solutions within {(good_range-1)*100:.2f}% of optimal')
#     print(f'{good_costs}')
#     if scen_filename is not None:
#         scen = pmu.Scenario(filename=scen_filename)
#         target_by_name = {str(_t.name):_t for _t in scen.targets}
#     def tg_seq_of_names(names):
#         target_names = [f'{_:d}' for _ in [int(_) for _ in names.split('-')]]
#         return [target_by_name[_tn] for _tn in target_names]
        
#     if add_best:
#         scen.add_solution('bestxx', best_dur, tg_seq_of_names(best_seqn))
#     if add_good:
#         for _i, (_d, _seqn) in enumerate(zip(good_costs, good_seqns)):
#             scen.add_solution(f'best{_i}', _d, tg_seq_of_names(_seqn))

#     #scen.fix()

#     if overwrite:
#         scen.save(scen_filename)
    
    
def main(id_scen=13):
    #test1(filename = 'scenario_30_2.yaml')
    #test2(filename = 'scenario_30_3.yaml')
    #test3()
    #play_anim('/tmp/scenario_60_3.yaml_199.77', 'heuristic_refined')


    if 0:
        plot_search_chronograms(filename = pmu.ressource('data/scenario_60_6_2.yaml'), epoch=int(2e4))

    if 0:
        data = create_or_load_search_set(pmu.ressource('data/scenario_60_6_2.yaml'),
                                          nb_searches=50, epochs=[int(5e3), int(1e4), int(2e4), int(5e4)],
                                          cache_filename=pmu.ressource('data/scenario_60_6_2_runs_50'),
                                          force_run=True)
    if 0:
        data = create_or_load_search_set(pmu.ressource('data/scenario_60_6_2.yaml'),
                                          nb_searches=10, epochs=[int(1e5)],
                                          cache_filename=pmu.ressource('data/scenario_60_6_2_runs_5'),
                                          force_run=True)        
    if 1:
        scen_filename = 'data/scenario_60_6_2.yaml'
        data = create_or_load_search_set(pmu.ressource(scen_filename),
                                          nb_searches=10, epochs=[int(1e4)],
                                          cache_filename=pmu.ressource('data/scenario_60_6_2_runs_50'),
                                          force_run=False)
    if 0: # concatenate files
        scen_filename = 'data/scenario_60_6_2.yaml'
        data = update_search_set(cache_filename1=pmu.ressource('data/scenario_60_6_2_runs_1'),
                                 cache_filename2=pmu.ressource('data/scenario_60_6_2_runs_2'),
                                 out_filename=pmu.ressource('data/scenario_60_6_2_runs'))
        for _f in ['data/scenario_60_6_2_runs_3', 'data/scenario_60_6_2_runs_4', 'data/scenario_60_6_2_runs_5', 'data/scenario_60_6_2_runs_6']:
            data = update_search_set(cache_filename1=pmu.ressource('data/scenario_60_6_2_runs'),
                                 cache_filename2=pmu.ressource(_f),
                                 out_filename=pmu.ressource('data/scenario_60_6_2_runs'))
    if 0:
        data = create_or_load_search_set(pmu.ressource('data/scenario_30_1.yaml'),
                                         nb_searches=20, epochs=[int(1e3), int(5e3), int(1e4)],
                                         cache_filename=pmu.ressource('data/scenario_30_1_runs'), force_run=False)

    if 0:
        data = create_or_load_search_set(pmu.ressource('data/scenario_15_3.yaml'),
                                         nb_searches=20, epochs=[int(1e3), int(5e3), int(1e4)],
                                         cache_filename=pmu.ressource('data/scenario_15_3_runs'), force_run=False)
    if 0:
        data = create_or_load_search_set(pmu.ressource('data/scenario_9_4.yaml'),
                                         nb_searches=50, epochs=[int(5e2), int(1e3), int(2e3), int(4e3)],
                                         cache_filename=pmu.ressource('data/scenario_9_3_runs'), force_run=False)
    analyze_search_set(*data, scen_filename=pmu.ressource('data/scenario_60_6_2.yaml'), add_best=False, add_good=False, overwrite=False)
    plot_search_set(*data, scen_filename)
        
    plt.show()
    
if __name__ == '__main__':
    main()
    
