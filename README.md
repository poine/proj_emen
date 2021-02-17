# proj_emen
Some drone catching targets planning thingy...

[Documentation index](https://poine.github.io/proj_emen/)

<figure class="cfigure">
	<img src="https://github.com/poine/proj_emen/blob/main/docs/images/anim_8_targets_fleeing.gif" alt="Interception examples." width="400">
	<figcaption>Fig0. -  Drone target interception thingy principle. Let's play sheepdog</figcaption>
</figure>
<br>

## Quick Start


### viewing a scenario (and solutions)

```
./display.py ../data/scen_30/1.yaml -a -nbest -f5
```

where `a` means animation, `best` is the name of the solution and `f` is for making annimation time run 5 times faster.


### performing one search

```
./search.py -m ex ../data/scen_small/8_1.yaml
```
performs an exhaustive search (tries all solutions). -m ex2 does the same in C++, it's faster

```
./search.py -m sa ../data/scen_30/1.yaml
```
performs a simulated anealing run. sa2 and sa3 are C++ equivalent



#### search sets


A tool for running and saving multiple searches.

```
./msearch.py -c -s ../data/scenario_15_6.yaml -t 10 -e1k,5k /tmp/scenario_15_6_runs_10
```
create the search


```
./msearch.py -x /tmp/scenario_15_6_runs_10 -a
```
displays histograms


#### parallel search

A bit like above but use python multiprocessing to run searches in parallel

```
./psearch.py ../data/scen_30/1.yaml  -e50k  -t12 -J8
```

This runs 12 simulated annealing searches, each 50k episodes. 
Searches will be run in parallel batches of 8. If unspecified, the number of available CPU cores will be used. 


#### Native code

install cython, move to src/native directory, type make, add the native directory to your LD_LIBRARY_PATH

Search methods `sa2`, `sa3` and `he2` require building the native code,  `he` and `sa` don't.
