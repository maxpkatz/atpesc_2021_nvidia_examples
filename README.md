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
