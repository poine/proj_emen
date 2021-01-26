---
title: Projet Emen
layout: default
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<script type="math/tex; mode=display">
    \newcommand{\vect}[1]{\underline{#1}}                         % vector
    \newcommand{\mat}[1]{\mathbf{#1}}                             % matrices
    \newcommand{\est}[1]{\hat{#1}}                                % estimate
    \newcommand{\err}[1]{\tilde{#1}}                              % error
    \newcommand{\esterr}[1]{\tilde{#1}}                           % error
    \newcommand{\pd}[2]{\frac{\partial{#1}}{\partial{#2}}}        % partial derivatives
    \newcommand{\transp}[1]{#1^{T}}                               % transpose
    \newcommand{\inv}[1]{#1^{-1}}                                 % invert
    \newcommand{\norm}[1]{|{#1}|}                                 % norm
    \newcommand{\esp}[1]{\mathbb{E}\left[{#1}\right]}             % expectation
    \newcommand{\identity}[0]{\mathbb{I}}                         % identity
    \newcommand{\jac}[3]{\frac{\partial{#1}}{\partial{#2}}|_{#3}} % Jacobian
</script>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>



## 1: Introduction

We consider a drone and a set of targets in an euclidian planar environment.
The targets are moving in a uniform rectilinear motion. The drone is moving at a constant forward velocity with a piecewise constant heading.
Our goal is to compute the optimal trajectory for the drone, i.e. the trajectory leading to the fastest interception of all targets as depicted on figure 0.

<figure class="cfigure">
  <img src="images/anim_8_targets_fleeing.gif" alt="Interception examples." width="255">
  <img src="images/anim_8_targets_approaching.gif" alt="Interception examples." width="255">
  <figcaption>Fig0. -  Drone target interception thingy principle. Let's play sheepdog</figcaption>
</figure>
<br>

## 2: Single target

### 2.1: Derivation
Here we establish the expression of the heading needed by the drone to intercept a single target.


The position of the drone and the target are given by:
{% comment %}  = \begin{pmatrix}x_{d0}+v_d \cos{\psi_d} t \\ y_{d0}+v_d \sin{\psi_d} t\end{pmatrix} {% endcomment %}
$$\begin{equation}
\vect{p}_d= \vect{p}_{d0} + \vect{v}_{d}.t
\label{eq:pos_drone}
\end{equation}
$$

{% comment %}
$$
\begin{pmatrix}v_{tx}\\v_{ty}\end{pmatrix}=\begin{pmatrix}v_t \cos{\psi_t}\\v_t \sin{\psi_t}\end{pmatrix}
$$

$$\begin{equation}
\vect{p}_t=\begin{pmatrix}x_{t0}+v_{tx} t \\ y_{t0}+v_{ty} t\end{pmatrix}
\label{eq:pos_target}
\end{equation}
$$
{% endcomment %}

$$\begin{equation}
\vect{p}_t= \vect{p}_{t0} + \vect{v}_{t}.t
\label{eq:pos_target}
\end{equation}
$$

At interception, using \eqref{eq:pos_drone} and \eqref{eq:pos_target}, we have:

$$
\vect{p}_d == \vect{p}_t  \iff \vect{p}_{d0} + \vect{v}_{d}.t = \vect{p}_{t0} + \vect{v}_{t}.t
$$

Rearanging, we get

$$\begin{equation}
t . \vect{\delta_v} = -\vect{\delta_{p0}}
\label{eq:intercept_cond}
\end{equation}
$$

where $$\vect{\delta_v} = \vect{v}_{d}-\vect{v}_{t} $$ and $$\vect{\delta_{p0}} = \vect{p}_{d0}-\vect{p}_{t0}$$ are the differences between drone and target velocities and initial positions.

Condition \eqref{eq:intercept_cond} imposes 

$$\begin{equation}
\vect{\delta_v} \wedge \vect{\delta_{p0}} = 0
\label{eq:intercept_cond1}
\end{equation}
$$


Expressing $$\vect{\delta_v}=\begin{pmatrix}v_d \cos{\psi_d} -v_{tx}\\v_d \sin{\psi_d} -v_{ty}\end{pmatrix}$$ and noting $$\vect{\delta_{p0}} = \begin{pmatrix}\delta_{p0x}\\\delta_{p0y}\end{pmatrix}$$, \eqref{eq:intercept_cond1} can be rewritten as

$$
(v_d \cos{\psi_d} -v_{tx})\delta_{p0x} - (v_d \sin{\psi_d} -v_{ty})\delta_{p0y} = 0
$$

or 

{% comment %}
$$
v_d \delta_{p0x} \cos{\psi_d} - v_d \delta_{p0y} \sin{\psi_d} = v_{tx}\delta_{p0x} - v_{ty}\delta_{p0y} 
$$

or
{% endcomment %}

$$\begin{equation}
a  \cos{\psi_d} + b \sin{\psi_d} = c
\label{eq:intercept_cond2}
\end{equation}
$$

with $$a=v_d \delta_{p0x}$$, $$b=-v_d \delta_{p0y}$$ and $$c=v_{tx}\delta_{p0x} - v_{ty}\delta_{p0y}$$


Substituting variable $$\lambda = \tan{\frac{\psi}{2}}$$ ($$\psi=2arctan{\lambda} $$), we get

$$
  \cos{\psi} = \frac{1-\lambda^2}{1+\lambda^2} \qquad \sin{\psi} = \frac{2\lambda}{1+\lambda^2}
$$

and \eqref{eq:intercept_cond2} becomes a second order polynomial

$$
(a+c) \lambda^2 -2b \lambda + (c-a) = 0
$$

which can be solved for its roots $$(\lambda_1, \lambda_2)$$, leading to a pair of headings $$(\psi_1, \psi_2)$$

Condition \eqref{eq:intercept_cond} can now be used once again to select the correct heading by enforcing

$$
\begin{equation}
\vect{\delta_v} . \vect{\delta_{p0}} < 0
\label{eq:intercept_cond3}
\end{equation}
$$

for $$t$$ positiveness.

$$t$$, the time of interception is then obtained from \eqref{eq:intercept_cond} as:

$$
t = \frac{|\vect{\delta_{p0}}|}{|\vect{\delta_v}|}
$$

#### Remark: Existence of a solution
    
As long as the forward velocity of the drone is strictly greater than the velocity of the target, a solution exists, by construction (TODO: show it, or just think about your chances of escaping when being chased by a faster guy...).

### 2.2: implementation

The above computation is implemented as follow 
<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fpoine%2Fproj_emen%2Fblob%2Ff65de3c1e3cd0c73b890d312f9791412f1fad86a%2Fsrc%2Fproj_manen.py%23L51-L63&style=github&showBorder=on&showLineNumbers=on"></script>

[code](https://github.com/poine/proj_emen/blob/main/src/proj_manen.py)


This [first test](https://github.com/poine/proj_emen/blob/main/src/test_1.py) runs our computation on a set of harcoded examples and displays the results as shown on figure 1. It additionally measures that my circa 2014 laptop is able to run this function at 10kHz.

<figure class="cfigure">
  <img src="images/single_interception_examples.png" alt="Interception examples." width="640">
  <figcaption>Fig1. -  Interception examples.</figcaption>
</figure>
<br>


## 3: Multiple targets


### 3.1: Implementation

When considering our initial problem, a set of targets, all we need to do is decide the sequence in which the interceptions will proceed.
With this information in hand, we apply our previous computation iteratively to the sequence of targets as follows:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fpoine%2Fproj_emen%2Fblob%2Ff65de3c1e3cd0c73b890d312f9791412f1fad86a%2Fsrc%2Fproj_manen.py%23L51-L69&style=github&showBorder=on&showLineNumbers=on"></script>

We start feeling the need of a way to store and describe scenarios, which we quench in the following way:

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fpoine%2Fproj_emen%2Fblob%2Ff9548a51450f6b4e163f28c710f764240a8b81ec%2Fsrc%2Fproj_manen.py%23L71-L81&style=github&showLineNumbers=on"></script>

[scenario_2.yaml](https://github.com/poine/proj_emen/blob/main/src/scenario_2.yaml)
<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fpoine%2Fproj_emen%2Fblob%2Fmain%2Fsrc%2Fscenario_2.yaml&style=github&showLineNumbers=on"></script>

We create a [6  targets scenari](https://github.com/poine/proj_emen/blob/main/src/scenario_6.yaml) with increasing headings and intercept them in order. This leads to the spiral like trajectory depicted on figure 2

<figure class="cfigure">
  <img src="images/one_sols_scen6.png" alt="Interception examples." width="640">
  <figcaption>Fig2. -  Interception examples with 6 targets.</figcaption>
</figure>
<br>

### 3.2: Exhaustive search

We create a [scenari with two targets](https://github.com/poine/proj_emen/blob/main/src/scenario_2.yaml) and test the two possible sequences as depicted on figure 3, one leading to a total time of $$3.08 s$$ and the other $$4.86 s$$
<figure class="cfigure">
  <img src="images/all_sols_scen2.png" alt="Interception examples." width="640">
  <figcaption>Fig3. -  Interception examples with 2 targets.</figcaption>
</figure>
<br>

We create a [scenari with three targets](https://github.com/poine/proj_emen/blob/main/src/scenario_3.yaml) and test the six possible sequences as depicted on figure 4. Total times vary between $$8.25s$$ and $$25.08s$$
<figure class="cfigure">
  <img src="images/all_sols_scen3.png" alt="Interception examples." width="640">
  <figcaption>Fig4. -  All sequences on a 3 targets example.</figcaption>
</figure>
<br>


On random scenario with increasing number of targets, we measure the time for an exhaustive search:


<figure class="cfigure">
  <img src="images/ex_search_time_vs_size.png" alt="Interception examples." width="640">
  <figcaption>Fig5. -  Exhaustive search time versus number of targets.</figcaption>
</figure>
<br>

It becomes increasingly clear that we will not be able to brute-force our way into this problem and that an exhaustive search becomes intractable for number of targets above 8.

### 3.3: Naïve heuristic


We create a naïve heuristic by selecting the target that is closest to the drone at each decision time.


<figure class="cfigure">
  <img src="images/anim_30_targets_heuristic_2.gif" alt="Interception examples." width="640">
  <figcaption>Fig6. -  Heuristics on 30 targets.</figcaption>
</figure>
<br>

As it seems to perform moderatly well on the large 30 targets example we throw at it, we seek to gain a quantitative evalution of its performances. We define a set of 7 targets scenarios

<figure class="cfigure">
  <img src="images/histo_scens_7.png" alt="histograms of 7 targets scenarios." width="640">
  <figcaption>Fig7. -  histograms of 7 targets scenarios.</figcaption>
</figure>
<br>

