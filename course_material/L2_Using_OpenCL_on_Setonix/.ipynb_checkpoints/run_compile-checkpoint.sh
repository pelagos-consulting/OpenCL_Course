#!/bin/bash -login

# Use a login shell to have a clean environment
module use /software/projects/courses01/setonix/opencl/modulefiles
module load PrgEnv-opencl
module load craype-accel-amd-gfx90a

# Run the make command
make clean
make