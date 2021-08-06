# NVIDIA Profiling Tutorial for ATPESC 2021

## Setup

First we need to make sure you've obtained access to a GPU on ThetaGPU.
We'll use an interactive session for this work.

```
module load cobalt/cobalt-gpu
qsub -n 1 -t 60 -I -q full-node
```

Now, we'll load the NVIDIA HPC SDK, which provides access to C, C++, and Fortran compilers.

```
module use /soft/thetagpu/hpc-sdk/modulefiles
module load nvhpc
```

## Introduction

In this lab we will focus on principles of performance analysis and optimization on NVIDIA GPUs, including the use of profiling tools to guide optimization.

A common motif in finite-element/finite-volume/finite-difference applications is the solution of elliptic partial differential equations with relaxation methods.
Perhaps the simplest elliptic PDE is the [Laplace equation](https://en.wikipedia.org/wiki/Laplace%27s_equation). The Laplace equation can be used to solve, for
example, the equilibrium distribution of temperature on a metal plate that is heated to a fixed temperature on its edges.

We want to solve this equation over a square domain that runs from 0 to L in both the x and y coordinates, given fixed boundary conditions at x = 0, x = L, y = 0,
and y = L. That is, we want to know what the temperature distribution looks like in the interior of the domain as a function of x. Characterizing the size of the
problem by N, a common approach is to discretize space into a set of N^2 points. If we denote the grid spacing as dx = dy = L / (N - 1), the points are located at
(0, 0), (dx, 0), (2 dx, 0), ..., (dy, 0), (2 dy, 0), ..., (L, L). If we treat the x coordinate with the index `i` and the y coordinate with the index `j`, then the
points with `i` = 0, `i` = N-1, `j` = 0, and `j` = N-1 remain fixed, while the interior (N-2)^2 points are the unknowns we need to solve for. For each index pair
(`i`, `j`) in the (zero-indexed) array, the coordinate position is (`i` dx, `j` dy)."

The discretized form of the Laplace equation for this case is:
```
    f_{i+1,j} - 2 f_{i,j} + f_{i-1,j} + f_{i,j+1} - 2 f_{i,j} + f_{i,j-1} = 0
```

Solving this for `f_{i,j}`, we get:
```
    f_{i,j} = (f_{i+1,j} + f_{i-1,j} + f_{i,j+1}, f_{i,j-1}) /  4
```

It turns out that we can simply *iterate* on this solution for f_{i,j} many times until the solution is sufficiently equilibrated. That is, if in every iteration
we take the old solution to be f, and then at every point in the new solution set it equal to the average of the four neighboring points from the old solution, we
will eventually solve for the equilibrium distribution of f. In (serial) pseudocode:
```
    // set initial conditions at f[0][:], f[N-1][:], f[:][0], f[:][N-1]

    while (error > tolerance):
        error = 0
        for i = 1, N-2:
            for j = 1, N-2:
                f[i][j] = 0.25 * (f_old[i+1][j] + f_old[i-1][j] + f_old[i][j+1] + f_old[i][j-1])
                error += (f[i][j] - f_old[i][j]) * (f[i][j] - f_old[i][j])
        swap(f_old, f)
```
