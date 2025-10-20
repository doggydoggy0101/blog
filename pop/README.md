# Polynomial Optimization

```sh
Example 1
min  -2x_1^2 - 2x_2^2 + 2x_1x_2 + 2x_1 + 6x_2 - 10
s.t. -x_1^2 + 2x_1 >= 0
     -x_1^2 - x_2^2 + 2x_1x_2 + 1 >=0
     -x_2^2 + 6x_2 - 8 >=0

2-th order moment relaxation status: optimal
Global optimality: True, rank: 3
Solution 1: [1. 2.]
Solution 2: [2. 3.]
Solution 3: [2. 2.]

Example 2
min 0
s.t.  x_1^2 + x_2^2 - 1 = 0
      x_1^3 + 2x_1x_2 + x_1x_2x_3 + x_2^3 - 1 = 0
      x_3^2 -2 = 0

2-th order moment relaxation status: optimal
Global optimality: False, rank: 8

3-th order moment relaxation status: optimal
Global optimality: False, rank: 6

4-th order moment relaxation status: optimal
Global optimality: True, rank: 6
Solution 1: [ 0.42857159  0.57142841 -0.64761913]
Solution 2: [-0.27803981 -0.72674834  3.78219867]
Solution 3: [0.07682401 1.1511649  0.9033483 ]
Solution 4: [ 0.12674753 -0.61570129 -0.14278444]
Solution 5: [-0.54740764  1.23863908 -1.61276566]
Solution 6: [ 1.19330449  0.38121705 -1.28237683]
```