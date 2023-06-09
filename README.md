# Accelerated computing with OpenCL

Open Compute Language (OpenCL) provides a programming framework for harnessing the compute capabilities of multicore processors. In this teaching series we cover a majority of the concepts for working with OpenCL in the context of supercomputing. We use the OpenCL 3.0 standard.

## Folder structure

* **course_material** - contains the course material.
* **deployment** - contains tools for deploying course material to Github and managing users.
* **resources** - helpful tools and information for use within the course.

## Syllabus

In this course we cover the following topics. Each topic is a subfolder in **course_material**.

* Lesson 1 - Introduction to OpenCL and high level features
* Lesson 2 - How to build and run OpenCL applications on Cray AMD systems like Frontier and Setonix
* Lesson 3 - A complete example of matrix multiplication, explained line by line
* Lesson 4 - Debugging OpenCL applications
* Lesson 5 - Measuring the performance of OpenCL applications with profiling and tracing tools
* Lesson 6 - Memory management
* Lesson 7 - Shared virtual memory
* Lesson 8 - Strategies for improving the performance of OpenCL kernels
* Lesson 9 - Strategies for optimising application performance with concurrent IO.

## Format

Lessons are in the form of Jupyter notebooks which can be viewed on the student's machine with JupyterLab or html files that can be viewed with a web browser. All exercises may be performed on the command line using an SSH connection to a remote server that has one or more implementations of OpenCL installed.

## Installation and dependencies

### Anaconda Python (optional)

A recent version of [Anaconda Python](https://www.anaconda.com/products/distribution) is helpful for viewing the notebook files. Once Anaconda is installed create a new environment 

```bash
conda create --name opencl_course python=3.10 nodejs=18.15.0
conda activate opencl_course
```
A list of helpful packages for viewing the material may then be installed with this command when run from the **OpenCL_Course** folder. 

```bash
pip install -r deployment/requirements.txt
```
then run 

```bash
jupyter-lab
```
from the command line to start the Jupyter Lab environment. The command

```bash
conda activate opencl_course
```
is to enter the created environment, and the command
```bash
conda deactivate
```
will leave the environment.

### Compiler

A C++ compiler must be installed and available through the **CC** command.

### OpenCL

Access to at least one OpenCL implementation is necessary. For teaching purposes it is preferred to have at least one CPU implementation and at least one GPU implementation installed, each with their `icd` files placed in **/etc/OpenCL/vendors**. The path to the OpenCL ICD loader **libOpenCL.so** must be in both **LIBRARY_PATH** and **LD_LIBRARY_PATH**. The location of the OpenCL headers, the **CL** directory (i.e where the file opencl.h lives) must be in **CPATH**. Khronos has a recent [ICD loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader) and [headers](https://github.com/KhronosGroup/OpenCL-Headers) that can be used to compile with and link against. Alternatively a fairly recent package for the OpenCL headers and ICD loader may be installed through a package manager.

#### OpenCL implementations

Here are some computing frameworks that contain OpenCL implementations. The trademark for each company is the property of their respective owners.

* [AMD ROCM](https://www.amd.com/en/graphics/servers-solutions-rocm)
* [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)
* [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#gs.zn3tzh)
* [Portable CL](http://portablecl.org/)

### Oclgrind

The debugging sections of this course use the excellent open source tool [Oclgrind](https://github.com/jrprice/Oclgrind). This can be installed through a package manager.

### Tau

Tau stands for [Tuning and Analysis Utilities](https://www.cs.uoregon.edu/research/tau/home.php). In the profiling section we use Tau to trace OpenCL calls.

### CLBlast

[CLBlast](https://github.com/CNugteren/CLBlast) is a linear algebra library we use to compare the performance of OpenCL kernels. This may be installed through a package manager.


