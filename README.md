# NVIDIA Profiling Tutorial for ATPESC 2021

## Setup

First we need to make sure you've obtained access to a GPU on ThetaGPU.
We'll use an interactive session for this work.

```
module load cobalt/cobalt-gpu
qsub -n 1 -t 60 -I -q full-node
```

In this lab we'll use CUDA 11.3 which is in the default ThetaGPU environment.
You can double check this with

```
nvcc --version
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
points with `i = 0`, `i = N-1`, `j = 0`, and `j = N-1` remain fixed, while the interior (N-2)^2 points are the unknowns we need to solve for. For each index pair
(`i`, `j`) in the (zero-indexed) array, the coordinate position is (`i` dx, `j` dy).

The discretized form of the Laplace equation for this case is:
```
f_{i+1,j} - 2 f_{i,j} + f_{i-1,j} + f_{i,j+1} - 2 f_{i,j} + f_{i,j-1} = 0
```

Solving this for `f_{i,j}`, we get:
```
f_{i,j} = (f_{i+1,j} + f_{i-1,j} + f_{i,j+1}, f_{i,j-1}) /  4
```

It turns out that we can simply *iterate* on this solution for `f_{i,j}` many times until the solution is sufficiently equilibrated. That is, if in every iteration
we take the old solution to be `f`, and then at every point in the new solution set it equal to the average of the four neighboring points from the old solution, we
will eventually solve for the equilibrium distribution of `f`. In (serial) pseudocode:
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

Let's walk through the process of converting a serial CPU code, jacobi.cpp, to run on a GPU. At each step in the optimization process we will stop and reflect, using
our developer tools to aid our understanding of the performance. First, verify that it runs on the CPU, noting the output (we periodically print the error).

```
g++ -o jacobi exercises/jacobi.cpp
./jacobi
```

## Step 1: Add NVTX Annotations

Before we jump into GPU porting, let's first identify where most of the time is being spent in our application. We're doing an extremely simple calculation, so you
likely have a good guess, but our philosophy here will be to measure performance carefully rather than assume we know how our application is porting (often, the
performance bottlenecks in your code will be surprising!). If you follow this methodical approach in your own application porting, you will likely succeed.

We could use standard CPU wall timers to profile our code, but instead we will choose to use a method that has better integration with the NVIDIA profiling tools: the
NVIDIA Tools Extension, or NVTX for short. NVTX is an instrumentation API that can be utilized by the NVIDIA developer tools to indicate sections of code with
human-readable string labels. Those labels can then be used in profiling output to demonstrate where time was spent.

The simplest NVTX API to use is the pair `nvtxRangePush()` and `nvtxRangePop()`, which add and remove a profiling region to a profiling region stack, respectively.
Using these two functions around a region of code looks like:
```
nvtxRangePush("my region name");
// do work here
nvtxRangePop();
```

Then the string "my region name" would be used in the profiling tool to time the section of code in between the push and the pop.

The only requirement for usage is that we include the right header (`nvToolsExt.h`) and link against the right runtime library (`libnvToolsExt.so`). These are located
in the `include/` and `lib64` directories of a CUDA installation.

So let's instrument the application with NVTX, being sure to mark off at least memory allocation, data initialization, the Jacobi relaxation step, and the data swap. (Note: NVTX
push/pop ranges can be nested, but we are avoiding that for now.) jacobi_step1.cpp contains the modified code with this instrumentation.

When you run the program under Nsight Systems, a region for NVTX ranges will appear at the end when using the stdout summary mode (`--stats=true`). What does the NVTX output
say about where the time is spent? Does it match your expectations?
```
g++ -o jacobi_step1 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 jacobi.cpp -lnvToolsExt
nsys profile --stats=true -o jacobi_step1 -f true ./jacobi_step1
```
