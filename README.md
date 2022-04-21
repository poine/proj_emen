# proj_emen

[Documentation index](https://poine.github.io/proj_emen/)

<figure class="cfigure">
	<img src="https://github.com/poine/proj_emen/blob/main/docs/images/anim_8_targets_fleeing.gif" alt="Interception examples." width="400">
	<figcaption>Figure 1: Drone target interception thingy general idea. Let's play sheepdog</figcaption>
</figure>
<br>

## Quick Start


### Viewing a scenario (and solutions)

```
./display.py ../data/scen_30/1.yaml -a -nbest -f5
```

where `a` means animation, `best` is the name of the solution and `f` is for making the animation run 5 times real time.


### Performing one search

```
./search.py -m ex ../data/scen_small/8_1.yaml
```
performs an exhaustive search (tries all solutions). -m ex2 does the same in C++, it's faster

```
./search.py -m sa ../data/scen_30/1.yaml
```
performs a simulated annealing run. sa2 and sa3 are C++ equivalent


### Parallel search

Same as above but uses python multiprocessing to run searches in parallel

```
./psearch.py ../data/scen_30/1.yaml  -e 50k  -t 12 -J 8
```

This runs 12 simulated annealing searches, each 50k episodes. 
Searches will be run in parallel batches of 8. If unspecified, the number of available CPU cores will be used. 
On my laptop, it runs in 1s and almost always returns what I believe to be the optimal solution (82.558s, [08-15-28-07-11...]).

### Tuning hyperparameters


A tool for running and saving multiple searches.

```
./tune_hp.py -c -s ../data/scen_small/15_6.yaml -t 10 -e 1k,5k /tmp/scen_15_6_runs_10.npz
```
runs and store the search results, ie 10 runs for 1k episodes and 10 runs for 5k episodes. 


```
./tune_hp.py -x /tmp/scen_15_6_runs_10.npz
```
load previously computed results and displays histograms



### Native code

install cython, move to src/native directory, run make, add the native directory to your LD_LIBRARY_PATH (so that libpm.so can be found)

Search methods `sa2`, `sa3` and `he2` require building the native code,  `he` and `sa` don't.


### Some videos

https://youtu.be/ra8ZkN9UhKE

https://youtu.be/g5M_gTPZ0AU

https://youtu.be/MNO4Uc2YVEI

https://youtu.be/rpfebwrCg4g

https://youtu.be/n1OcXBNmzlQ



