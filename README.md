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
## Step 5: Convert the Swap Kernel

We saw above that the Jacobi kernel got significantly faster in absolute terms, though perhaps not as much as we would have hoped, but it made the swap kernel slower! It appears
that we're paying both for the cost of Unified Memory transfers from host to device in the Jacobi kernel, and Unified Memory transfers from device to host when the swap function occurs.

At this point we have two options to improve performance. We could either use Unified Memory prefetches to move the data more efficiently, or we could just go ahead and port the
swap kernel to CUDA as well. We are going to suggest the latter. Our goal should be to do as much of the compute work as possible on the GPU, and in this case it's entirely possible
to keep the data on the GPU for the entirety of the Jacobi iteration.

So now we implement `swap_data()` in CUDA and check the profiler output to understand what happened. See `jacobi_step5.cpp` for the new code.
```
nvcc -o jacobi_step5 -x cu -arch=sm_80 -lnvToolsExt jacobi_step5.cpp
nsys profile --stats=true -o jacobi_step5 -f true ./jacobi_step5
```

## Step 6: Analyze the Jacobi Kernel

We are now much faster on the GPU than on the CPU, and from our profiling output we should now be able to identify that the vast majority of the application runtime is spent in kernels,
particularly the Jacobi kernel (the swap kernel apears to be very fast in comparison). It is now appropriate to start analyzing the performance of this kernel and ask if there are any
optimizations we can apply.

First, let us hypothesize about whether an ideal implementation of this kernel should be compute-bound, memory-bound, or neither (latency-bound). To avoid being latency-bound, we should
generally expose enough work to keep a large number of threads running on the device. If `N = 2048`, say, then there are `2048 * 2048` or about 4 million degrees of freedom in this problem.
Since the order of magnitude number of threads a modern high-end GPU can simultaneously field is O(100k), we likely do enough work to keep the device busy -- though we will have to verify this.

When thinking about whether we are compute-bound or memory-bound, it is natural to think in terms of the *arithmetic intensity* of the operation, that is, the number of (floating-point)
operations computed per byte moved. A modern accelerator typically is only compute bound when this ratio is of order 10. But the Jacobi stencil involves moving four words (each of which is
4 bytes, for single-precision floating point) while only computing four floating point operations (three adds and one multiply). So the arithmetic intensity is 4 (FLOPs) / 16 (bytes) = 0.25
FLOPs / byte. Clearly this is in the memory-bandwidth bound regime.

With that analysis in mind, let's see what the profiling tool tells us about the memory throughput compared to speed-of-light. We'll apply Nsight Compute to the program we compiled in Step 5.
We assume that every invocation of the kernel has approximately similar performance characteristics, so we only profile one invocation, and we skip the first few to allow the device to warm up.
We'll save the input to a file first, and then import the results to display in the terminal (in case we want to open the report in the Nsight Compute user interface).
```
ncu --launch-count 1 --launch-skip 5 --kernel-regex jacobi --export jacobi_step5 --force-overwrite ./jacobi_step5
ncu --import jacobi_step5.ncu-rep
```

You likely saw that, as predicted, we have pretty decent achieved occupancy, so we're giving the GPU enough work to do, but we are definitely not close to speed-of-light on compute throughput
(SM %), and disappointingly we're also not even close on memory throughput either. We appear to still be latency-bound in some way.

A likely culprit explaining low memory throughput is poor memory access patterns. Since we're currently working with global memory, that usually implies uncoalesced memory accesses. Could that
have anything to do with it?

If that's where we're headed, then at this point an obvious question to ask is, does our threading strategy actually map well to the memory layout of our arrays? This can be a tricky thing to
sort out when working with 2D data and 2D blocks, and it gets even more complicated for the Jacobi kernel in particular because of the stencil pattern. However, we fortunately have a kernel that
we know uses exactly the same threading strategy (`swap_data()`) that is much faster. What does Nsight Compute say about the memory throughput of that kernel? We can use the `--set full` to do
a more through (albeit more expensive) analysis. At the bottom of the output, Nsight Compute will tell us if there were a significant amount of uncoalesced accesses.
```
ncu --launch-count 1 --launch-skip 5 --kernel-regex swap_data --set full ./jacobi_step5
```

OK, so we can clearly conclude two things based on this output. First, there are plenty of uncoalesced accesses in this kernel -- in fact, Nsight Compute tells us that we did 8x as many sector
loads as we needed (sectors are the smallest unit of accessible DRAM in a memory transaction and are 32-bytes long)! This is a smoking gun for a memory access pattern with a large stride.
Correspondingly, we only got a small fraction of DRAM throughput. Second, despite this fact, we achieved a pretty decent fraction of speed-of-light for *L2 cache* accesses. This makes sense
to the extent that caching is helping to ameliorate the poor nature of our DRAM access pattern, but a question to ask yourself is, why didn't the Jacobi kernel achieve that? We'll come back
to that later.

Compare the indexing scheme to the threading scheme, noting that in a two-dimensional threadblock, the `x` dimension is the contiguous dimension and the `y` dimension is the strided dimension;
in a 32x32 thread block, you can think of `threadIdx.y` as enumerating which of 32 warps we're using with, while each warp constitutes the 32 threads in the `x` dimension. So if we want to fix
our memory access pattern we can reverse our indexing macro to effectively reverse the memory layout: the `i` dimension will be contiguous in memory to match the "shape" of the threadblock.
This fix will allow us to achieve coalesced memory accesses. Look at `jacobi_step6.cpp` for this change.

As a side note, we expect this to improve GPU performance, but it's also likely this would have improved the CPU performance as well (if we compare to the original code where the `i` dimension
was the innermost loop), which will make the overall GPU speedup relative to the CPU less impressive. You may want to go back and check how much of a factor that was in the CPU-only code.
```
nvcc -o jacobi_step6 -x cu -arch=sm_80 -lnvToolsExt jacobi_step6.cpp
nsys profile --stats=true -o jacobi_step6 -f true ./jacobi_step6
```

Verify using Nsight Compute that the DRAM throughput of the swap kernel is better now.
```
ncu --launch-count 1 --launch-skip 5 --kernel-regex swap_data --set full ./jacobi_step6
```

## Step 7: Make the Problem Bigger (Again)

The DRAM throughput is not quite where we want it. `swap_data()` isn't even using half of DRAM throughput. Let's make the problem bigger again! See `jacobi_step7.cpp`.
```
nvcc -o jacobi_step7 -x cu -arch=sm_80 -lnvToolsExt jacobi_step7.cpp
nsys profile --stats=true -o jacobi_step7 -f true ./jacobi_step7
```

```
ncu --launch-count 1 --launch-skip 5 --kernel-regex swap_data --set full ./jacobi_step7
```

## Step 8: Revisiting the Reduction

Let's take another look at the Jacobi kernel now that we've fixed the overall global memory access pattern and have a big enough problem to saturate DRAM throughput.
```
ncu --launch-count 1 --launch-skip 5 --kernel-regex jacobi --export jacobi_step7 --force-overwrite --set full ./jacobi_step7
ncu --import jacobi_step7.ncu-rep
```

The output from Nsight Compute is a little puzzling here. We know we probably have some work to do with our stencil operation, but surely we should be doing better than ~1% of peak memory
throughput? If we consult the Scheduler Statistics section, we see that a shocking 99.9% of cycles have no warps eligible to issue work! This is much, much worse than the swap kernel, which
has eligible warps on average 15% of the time. What could explain that? Well, the only thing we're really doing in our kernel besides the stencil update is the atomic update to the error
counter. And we have reason to believe that if many threads are writing atomically to the same location at the same time, they will serialize and stall. In the beginning, we did it this way
because just getting the work on the GPU to begin with was the clear first step, but now we obviously need to revisit this.

Let's refactor the kernel to use a more efficient reduction scheme that uses fewer overall atomics. We'll use the NVIDIA library [cub](https://nvlabs.github.io/cub/). To get a sense of how
good of a job you would like to do, try commenting out the atomic reduction entirely from the kernel (and then temporarily modifying `main()` so that you can run enough iterations of the
Jacobi kernel to get a profiling result, and see how much faster it is in that case (and inspect the Nsight Compute output for that case). That's your "speed of light" kernel, at least
with respect to the reduction phase. Our new code is in `jacobi_step8.cpp`.
```
nvcc -o jacobi_step8 -x cu -arch=sm_80 -lnvToolsExt jacobi_step8.cpp
nsys profile --stats=true -o jacobi_step8 -f true ./jacobi_step8
```

Make sure to check the Nsight Compute output to see how close we get to the DRAM throughput of the case with no reduction at all.
```
ncu --launch-count 1 --launch-skip 5 --kernel-regex jacobi --export jacobi_step8 --force-overwrite --set full ./jacobi_step8
ncu --import jacobi_step8.ncu-rep
```

## Step 9: Shared Memory

Similar to the thought experiment we did about how fast our reduction ought to be, we can also do a thought experiment about what the speed of light for the Jacobi kernel is. We have a
nice comparison kernel in `swap_data()`, which has fully coalesced accesses. Since there is a significant delta in DRAM throughput between that kernel and the Jacobi kernel, we'd like
to see if there's anything we can do.

Stencil operations are non-trivial for GPU caches to deal with properly because the very large stride between row `j` and row `j+1` (corresponding to the number of columns) means that we may
get little cache reuse, a problem that only grows as the array size becomes larger. This is a good candidate for caching the stencil data in shared memory, so that when we do reuse the
data, it's reading from a higher bandwidth, lower latency source close to the compute cores. Note that we're not trying to improve DRAM throughput, we're trying to access DRAM less frequently.

Let's implement shared memory usage in the kernel. We'll use a 2D tile. For simplicity, we assume that the number of threads in the block is 1024 and that there are 32 per dimension (we can't
get larger than this anyway), so that we can hardcode the size of the shared memory array at compile time. There's a couple ways to do this; the simplest way, and the one we recommend, is simply
to read in the data into a 2D tile whose extent is 34x34 (since we need to update 32x32 values and the stencil depends on data up to one element away in each dimension). We make sure to perform
a proper threadblock synchronization before reading the data in shared memory, and we note that the `__syncthreads()` intrinsic needs to be called by all threads in the block.

A solution to this is presented in `jacobi_step9.cpp`. Note that this is probably not the best possible solution, it errs slightly on the side of legibility over performance in how the
shared memory tile is loaded.
```
nvcc -o jacobi_step9 -x cu -arch=sm_80 -lnvToolsExt jacobi_step9.cpp
nsys profile --stats=true -o jacobi_step9 -f true ./jacobi_step9
```

```
ncu --launch-count 1 --launch-skip 5 --kernel-regex jacobi --export jacobi_step9 --force-overwrite --set full ./jacobi_step9
ncu --import jacobi_step9.ncu-rep
```

## Closing Thoughts

After all of these steps, the kernels are now so fast again that the device warmup may be again a salient performance factor. In this case, we may want to again consider increasing the size of
the problem to amortize this cost out. If you do, try comparing it to the CPU implementation to see what our final speedup was.
