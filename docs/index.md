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


## 2: Single target

### 2.1: Derivation

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

$$
v_d \delta_{p0x} \cos{\psi_d} - v_d \delta_{p0y} \sin{\psi_d} = v_{tx}\delta_{p0x} - v_{ty}\delta_{p0y} 
$$

or

$$\begin{equation}
a  \cos{\psi_d} + b \sin{\psi_d} = c
\label{eq:intercept_cond2}
\end{equation}
$$

Substituting variable $$\lambda = \tan{\frac{\psi}{2}}$$ ($$\psi=2arctan{\lambda} $$), we get

$$
  \cos{\psi} = \frac{1-\lambda^2}{1+\lambda^2} \qquad \sin{\psi} = \frac{2\lambda}{1+\lambda^2}
$$

and \eqref{eq:intercept_cond2} becomes a second order polynomial

$$
(a+c) \lambda^2 -2b \lambda + (c-a) = 0
$$

which can be solved for its two root $$\lambda_1$$ and $$\lambda_2$$, leading to a pair of headings $$\psi_1$$ and $$\psi_2$$

Condition \eqref{eq:intercept_cond} can now be used once again to select the correct heading by enforcing

$$
\begin{equation}
\vect{\delta_v} . \vect{\delta_{p0}} < 0
\label{eq:intercept_cond3}
\end{equation}
$$
 because $$t$$ is positive.

### 2.2: implementation

The above computation is implemented as follow 

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fpoine%2Fproj_emen%2Fblob%2F3e149a03fe32268cae30528b071e3b222d285292%2Fsrc%2Fproj_manen.py%23L52-L63&style=github&showBorder=on&showLineNumbers=on"></script>

[code](https://github.com/poine/proj_emen/blob/3e149a03fe32268cae30528b071e3b222d285292/src/proj_manen.py#L52-L63)


This [first test](bla) runs our computation on a set of harcoded examples and displays the results as shown on figure 1

<figure class="cfigure">
  <img src="images/single_interception_examples.png" alt="Interception examples." width="640">
  <figcaption>Fig1. -  Interception examples.</figcaption>
</figure>
<br>


## 3: Multiple targets

We extends our above behaviour by 



All we have decide is the order in which we deal with the targets


<figure class="cfigure">
  <img src="images/one_sols_scen6.png" alt="Interception examples." width="640">
  <figcaption>Fig1. -  Interception examples.</figcaption>
</figure>
<br>


<figure class="cfigure">
  <img src="images/all_sols_scen2.png" alt="Interception examples." width="640">
  <figcaption>Fig1. -  Interception examples.</figcaption>
</figure>
<br>

<figure class="cfigure">
  <img src="images/all_sols_scen3.png" alt="Interception examples." width="640">
  <figcaption>Fig1. -  Interception examples.</figcaption>
</figure>
<br>

