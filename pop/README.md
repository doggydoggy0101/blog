# Polynomial Optimization
Python implementation of Lasserre's hierarchy for polynomial optimization problems. For more information, please refer to my blog: [Lasserre Hierarchy](https://dgbshien.com/docs/blogs/lasserre-hierarchy.pdf).

**Prerequisites**
```sh
pip install numpy scipy sympy "cvxpy[mosek]"
```

We test examples from the paper [Detecting Global Optimality and Extracting Solutions in GloptiPoly](https://homepages.laas.fr/henrion/papers/extract.pdf).

**Example in Section 2.3**

$$
\begin{aligned}
\min_{x\in\mathbb{R}^2}\ &-(x_1-1)^2-(x_1-x_2)^2-(x_2-3)^2\\
\text{s.t.}\ &\ 1-(x_1-1)^2\geq0\\
&\ 1-(x_1-x_2)^2\geq0\\
&\ 1-(x_2-1)^2\geq0\\
\end{aligned}
$$

```shell
[POP] rank diff condition: 1

[POP] 2-th order moment relaxation:
[POP] sdp optimality: True
[POP] rank(M_1)=3
[POP] rank(M_2)=3
[POP] pop optimality: True

value: -2.0
solution 1: [2. 3.]
solution 2: [1. 2.]
solution 3: [2. 2.]
```

**Example in Section 3.1**

$$
\begin{aligned}
x_1^2+x_2^2 -1&=0\\
x_1^3+(2+x_3)x_1x_2+x_2^3 -1&=0\\
x_3^2-2 &=0
\end{aligned}
$$

```shell
[POP] No objective found, minimize the trace of moment matrix
[POP] rank diff condition: 2

[POP] 2-th order moment relaxation:
[POP] sdp optimality: True
[POP] rank(M_1)=4
[POP] rank(M_2)=7
[POP] pop optimality: False

[POP] 3-th order moment relaxation:
[POP] sdp optimality: True
[POP] rank(M_1)=2
[POP] rank(M_2)=2
[POP] rank(M_3)=2
[POP] pop optimality: True

value: 24.75
solution 1: [ 0.7071  0.7071 -1.4142]
solution 2: [-0.7071 -0.7071  1.4142]
```

**Example in Section 3.2**

$$
\begin{aligned}
5x_1^9-6x_1^5x_2 + x_1x_2^4 + 2x_1x_3 &= 0\\
-2x_1^6x_2 + 2x_1^2x_2^3 + 2x_2x_3 &= 0\\
x_1^2 + x_2^2 &= 0.265625
\end{aligned}
$$

```shell
[POP] No objective found, minimize the trace of moment matrix
[POP] rank diff condition: 5

[POP] 5-th order moment relaxation:
[POP] sdp optimality: True
[POP] rank(M_1)=3
[POP] rank(M_2)=4
[POP] rank(M_3)=4
[POP] rank(M_4)=4
[POP] rank(M_5)=4
[POP] pop optimality: False

[POP] 6-th order moment relaxation:
[POP] sdp optimality: True
[POP] rank(M_1)=2
[POP] rank(M_2)=2
[POP] rank(M_3)=2
[POP] rank(M_4)=2
[POP] rank(M_5)=2
[POP] rank(M_6)=2
[POP] pop optimality: True

value: 1.3373
solution 1: [ 0.2619  0.4439 -0.0132]
solution 2: [-0.2619  0.4439 -0.0132]
```