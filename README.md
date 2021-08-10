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
push/pop ranges can be nested, but we are avoiding that for now.) `jacobi_step1.cpp` contains the modified code with this instrumentation.

When you run the program under Nsight Systems, a region for NVTX ranges will appear at the end when using the stdout summary mode (`--stats=true`). What does the NVTX output
say about where the time is spent? Does it match your expectations?
```
g++ -o jacobi_step1 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 jacobi_step1.cpp -lnvToolsExt
nsys profile --stats=true -o jacobi_step1 -f true ./jacobi_step1
```

## Step 2: Unified Memory

Rather than using standard CPU `malloc()`, we can use `cudaMallocManaged()` to allocate our data in Unified Memory (but we don't make any other changes yet).
This is completely legal even if, as in this case, we only intend to use host code (for now). We also add CUDA error checking. These changes require us to add
the API `cuda_runtime_api.h` and the runtime library `libcudart.so`. What does the profile indicate about the relative cost of starting up a CUDA program?
The new code is in `jacobi_step2.cpp`.
```
g++ -o jacobi_step2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 jacobi_step2.cpp -lnvToolsExt -lcudart
nsys profile --stats=true -o jacobi_step2 -f true ./jacobi_step2
```

## Step 3: Make the Problem Bigger

In step 2, we saw that the cost of initializing CUDA (called "context creation") can be high, often measured in the hundreds of milliseconds. In this case,
the cost is so large that it dwarfs the time spent in the actual calculation. Even if we could make the cost of the calculation zero with infinitely fast kernels,
we would only make a small dent in the runtime of the application, and it would still be much slower than the original CPU-only calculation (without Unified Memory) was.
There is simply no sense trying to optimize this scenario.

When faced with this problem, the main conclusion to reach is that you need to *solve a bigger problem*. In many scientific applications there are two primary ways to
make the problem bigger: we can either add more elements/zones/particles, or we can increase the number of iterations the code runs. In this specific case, the options
are to increase the number of points in the grid, or to use a stricter error tolerance (which should require more iterations to achieve). However, if you make the tolerance
several orders of magnitude tighter, you will only increase the number of steps by a relatively small factor for this particular case. (Most of the work is in transforming
from our terrible initial guess of all zeros to a state that approximates the correct solution; the rest is fine-tuning.) So we have to use more grid points, which will
achieve a finer spatial resolution and thus a more accurate (but expensive) answer. **This is a general fact of life when using GPUs: often it only makes sense to solve a
much more expensive problem than the one we were solving before on CPUs.**

So let's increase the number of grid points, `N`, such that the time spent in the main relaxation phase is at least 95% of the total application time. For the purpose of
simplicity later, we are keeping it as a factor of 2, and we recommend a value of at least `N = 2048`. The updated code is in `jacobi_step3.cpp`.
```
g++ -o jacobi_step3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 jacobi_step3.cpp -lnvToolsExt -lcudart
nsys profile --stats=true -o jacobi_step3 -f true ./jacobi_step3
```

## Step 4: Run the Jacobi Step on the GPU

The Jacobi relaxation steps are now the most expensive portion of the application. We also know this is a task we can solve in parallel since each zone is updated independently
(with the exception of the error, for which we must perform some sort of reduction). We first convert `jacobi_step()` (only) to a CUDA kernel. We parallelize over both the inner
and outer loop using a two-dimensional threadblock of size (32x32), so that the body of the function doesn't contain any loops, just the update to `f` and the `error`. How much
faster does the Jacobi step get? How much faster does the application get overall? What is the new application bottleneck? The new code is in `jacobi_step4.cpp`. Note that we now
need to use `nvcc` since we're compiling CUDA code.
```
nvcc -o jacobi_step4 -x cu -arch=sm_80 -lnvToolsExt jacobi_step4.cpp
nsys profile --stats=true -o jacobi_step4 -f true ./jacobi_step4
```
