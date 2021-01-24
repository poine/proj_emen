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


# Index

## Introduction


## Single target

Drone

$$\begin{equation}
\vect{p}_d=\begin{pmatrix}x_{d0}+v_d \cos{\psi_d} t \\ y_{d0}+v_d \sin{\psi_d} t\end{pmatrix}
\label{eq:pos_drone}
\end{equation}
$$

Target

$$
\begin{pmatrix}v_{tx}\\v_{ty}\end{pmatrix}=\begin{pmatrix}v_t \cos{\psi_t}\\v_t \sin{\psi_t}\end{pmatrix}
$$

$$\begin{equation}
\vect{p}_t=\begin{pmatrix}x_{t0}+v_{tx} t \\ y_{t0}+v_{ty} t\end{pmatrix}
\label{eq:pos_target}
\end{equation}
$$

\eqref{eq:pos_drone}
\eqref{eq:pos_target}

$$
\vect{p}_d == \vect{p}_t  \iff 
\begin{cases}
x_{d0}+v_d \cos{\psi_d} t = x_{t0}+v_{tx} t\\
y_{d0}+v_d \sin{\psi_d} t = y_{t0}+v_{ty} t
\end{cases}
$$

Solving for t, we get

$$
\begin{cases}
t(v_d \cos{\psi_d}-v_{tx})= x_{t0} - x_{d0}\\
t(v_d \sin{\psi_d}-v_{ty})= y_{t0} - y_{d0}
\end{cases}
$$

<script src="https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2Fpoine%2Fproj_emen%2Fblob%2F3e149a03fe32268cae30528b071e3b222d285292%2Fsrc%2Fproj_manen.py%23L52-L63&style=github&showBorder=on&showLineNumbers=on"></script>

[code](https://github.com/poine/proj_emen/blob/3e149a03fe32268cae30528b071e3b222d285292/src/proj_manen.py#L52-L63)

<figure class="cfigure">
  <img src="images/single_interception_examples.png" alt="Interception examples." width="640">
  <figcaption>Fig1. -  Interception examples.</figcaption>
</figure>
<br>


## Multiple targets

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




```ruby
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```



{% comment %}
  * [Second Order LTI](so_lti.html)
{% endcomment %}


