"""@package py_helper
Helper functions to facilitate loading and displaying results.
 
Written by Dr. Toby Potter 
for the Commonwealth Scientific and Industrial Research Organisation of Australia (CSIRO).
"""

import numpy as np
import ast
import math
from matplotlib import pyplot as plt
from ipywidgets import widgets

# Import axes machinery
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
from collections.abc import Iterable
from collections import OrderedDict
import json

def load_defines(fname):
    """Load all defines from a header file"""
    defines = {}
    with open(fname, "r") as fd:
        for line in fd:
            if line.startswith("#define"):
                tokens=line.split()
                defines[tokens[1]]=ast.literal_eval(tokens[2])
    return defines

class MatMul:
    """Implements the means to define and test and run a matrix multiplication"""
    def __init__(self, NCOLS_A, NROWS_C, NCOLS_C, dtype):
        """Constructor for the class"""
        self.NCOLS_A = NCOLS_A
        self.NROWS_C = NROWS_C
        self.NCOLS_C = NCOLS_C
        self.dtype = dtype
  
    def run_compute(self):
        """Calculate the solution"""
        self.C = np.matmul(self.A, self.B, dtype = self.dtype)

    def load_data(self):
        """Load binary arrays from file"""
        self.A = np.fromfile("array_A.dat", dtype=self.dtype).reshape((self.NROWS_C, self.NCOLS_A))
        self.B = np.fromfile("array_B.dat", dtype=self.dtype).reshape((self.NCOLS_A, self.NCOLS_C))
        self.run_compute()

    def make_data(self):
        """Make up the arrays A, B, and C"""
        
        # A is of size (NROWS_C, NCOLS_A)
        # B is of size (NCOLS_A, NCOLS_C)    
        # C is of size (NROWS_C, NCOLS_C)
        
        self.A = np.random.random(size = (self.NROWS_C, self.NCOLS_A)).astype(self.dtype)
        self.B = np.random.random(size = (self.NCOLS_A, self.NCOLS_C)).astype(self.dtype)

        # Make up the answer
        self.run_compute()

        # Write out the arrays as binary files
        self.A.tofile("array_A.dat")
        self.B.tofile("array_B.dat")

    def check_data(self):
        """Load data from file and check against the computed solution"""
        
        # Make sure we have the solution
        assert hasattr(self, "C"), "Must run make_data() before check_data()."
        
        # Read in the output from file
        self.C_out = np.fromfile("array_C.dat", dtype=self.dtype).reshape((self.NROWS_C, self.NCOLS_C))

        # Make plots
        fig, axes = plt.subplots(3, 1, figsize=(6,8), sharex=True, sharey=True)

        # Data to plot
        data = [self.C, self.C_out, np.abs(self.C-self.C_out)]

        # Labels to plot
        labels = ["Numpy", "Program", "Absolute residual"]

        for n, value in enumerate(data):
            # Plot the graph
            ax = axes[n]
            im = ax.imshow(value)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            # Set labels on things
            ax.set_xlabel("Dimension 1 (columns)")
            ax.set_ylabel("Dimension 0 (rows)")
            ax.set_title(labels[n])

            # Put a color bar on the plot
            plt.colorbar(mappable=im, cax=cax)

        fig.tight_layout()
        plt.show()
        
class Hadamard:
    """Implements the means to define, test, and run an elementwise (Hadamard)
    matrix multiplication."""
    def __init__(self, NROWS_F, NCOLS_F, dtype):
        self.NROWS_F = NROWS_F
        self.NCOLS_F = NCOLS_F
        self.dtype = dtype
        
    def run_compute(self):
        """Compute the transformation."""
        self.F = self.D*self.E

    def load_data(self):
        """Load the output data from file."""
        self.D = np.fromfile("array_D.dat", dtype=self.dtype).reshape((self.NROWS_F, self.NCOLS_F))
        self.E = np.fromfile("array_E.dat", dtype=self.dtype).reshape((self.NROWS_F, self.NCOLS_F))
        self.run_compute()

    def make_data(self):
        """Make up the solution and write it out to file."""
    
        # D is of size (NROWS_F, NCOLS_F)
        # E is of size (NCOLS_F, NCOLS_F)    
        # F is of size (NROWS_F, NCOLS_F)

        # Make up the arrays A, B, and C
        self.D = np.random.random(size = (self.NROWS_F, self.NCOLS_F)).astype(self.dtype)
        self.E = np.random.random(size = (self.NROWS_F, self.NCOLS_F)).astype(self.dtype)

        # Make up the answer using Hadamard multiplication
        self.run_compute()

        # Write out the arrays as binary files
        self.D.tofile("array_D.dat")
        self.E.tofile("array_E.dat")

    def check_data(self):
        """Load a binary file and check it against the local solution"""
        
        # Make sure we have the solution
        assert hasattr(self, "F"), "Must run make_data() or load_data before check_data()."

        # Read in the output from OpenCL
        self.F_out = np.fromfile("array_F.dat", dtype=self.dtype).reshape((self.NROWS_F, self.NCOLS_F))

        # Make plots
        fig, axes = plt.subplots(3, 1, figsize=(6,8), sharex=True, sharey=True)

        # Data to plot
        data = [self.F, self.F_out, np.abs(self.F-self.F_out)]

        # Labels to plot
        labels = ["Numpy", "Program", "Absolute residual"]

        for n, value in enumerate(data):
            # Plot the graph
            ax = axes[n]
            im = ax.imshow(value)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

            # Set labels on things
            ax.set_xlabel("Dimension 1 (columns)")
            ax.set_ylabel("Dimension 0 (rows)")
            ax.set_title(labels[n])

            # Put a color bar on the plot
            plt.colorbar(mappable=im, cax=cax)

        fig.tight_layout()
        plt.show()
        
class LocalOpt():
    """Class to capture the result of an optimisation exercise for different algorithms."""
    
    def __init__(self, 
            timings=None, 
            cmds=None, 
            local0=np.uint32(2**np.arange(0,10,1)),
            local1=np.uint32(2**np.arange(0,10,1)),
            local2=np.uint32(2**np.arange(0,1,1))
                ):
        
        # Does the class have any data?
        self.has_data=False
        
        if timings is not None:
            assert(cmds is None)
            self.import_result(timing_data)
            
        if cmds is not None:
            assert(timings is None)
            self.make_result(cmds, local0, local1, local2)
        
    def make_mesh(self, local0, local1, local2):
        return np.meshgrid(local0, local1, local2, indexing="ij") 
        
    def insert_local(self, local0, local1, local2, times_ms, times_stdev):
        
        # Set variables
        self.local0 = np.array(local0)
        self.local1 = np.array(local1)
        self.local2 = np.array(local2)
            
        self.L0, self.L1, self.L2 = self.make_mesh(local0, local1, local2)
            
        # Data to plot
        self.times_ms = np.array(times_ms)
        self.times_stdev = np.array(times_stdev)

        # Signal that we have data
        self.has_data=True 
          
    def import_result(self, timing_data):
        self.insert_local(
            timing_data["local0"],
            timing_data["local1"],
            timing_data["local2"],
            timing_data["times_ms"],
            timing_data["times_stdev"])
        
    def report_timings(self, timing_data):
        """Find the minimum and maximum of timing data obtained from an experiement"""
        
        print(f"Min time is {timing_data['min_ms']:.3f} ms, at the local size of" 
            f" ({timing_data['L0_min']},{timing_data['L1_min']},{timing_data['L2_min']}).")
        print(f"Max time is {timing_data['max_ms']:.3f} ms, at the local size of" 
            f" ({timing_data['L0_max']},{timing_data['L1_max']},{timing_data['L2_max']}).")
        print(f"Max time / min time == {timing_data['max_ms']/timing_data['min_ms']:.3f}")
  
    def make_result(self, 
                    cmds,
                    # Vectors of local sizes
                    local0=np.uint32(2**np.arange(0,10,1)),
                    local1=np.uint32(2**np.arange(0,10,1)),
                    local2=np.uint32(2**np.arange(0,1,1))):
        
        """Prepare the input file for local optimisation and run the problem"""
        
        # Make up the optimisation grid
        L0, L1, L2 = self.make_mesh(local0, local1, local2)   
    
        input_local = np.zeros((*L0.shape, 3), dtype=np.uint32)
        input_local[...,0] = L0
        input_local[...,1] = L1
        input_local[...,2] = L2
        
        # Write out to file
        input_local.tofile("input_local.dat")        
    
        # Add the --local-file flag
        if isinstance(cmds, Iterable) and not isinstance(cmds, str):
            temp_cmds = list(cmds) + ["--local_file"]
        else:
            temp_cmds=[cmds,"--local_file"]
        
        # Run the program 
        result = subprocess.run(temp_cmds)
        print(f"returncode is {result.returncode}")
        
        if (result.returncode==0):
        
            # Get the output data
            output_local = np.fromfile("output_local.dat", dtype=np.float64).reshape(
                local0.size, local1.size, local2.size, 2, order="C"
            )
            
            # Data to plot
            times_ms = output_local[...,0]
            times_stdev = output_local[...,1]
            
            self.insert_local(local0, local1, local2, times_ms, times_stdev)
   
    def export_result(self):
        
        assert(self.has_data==True)
            
        # Find the minimum time
        index_min = np.nanargmin(self.times_ms)
        index_max = np.nanargmax(self.times_ms)

        timing_data = {
            "min_ms" : self.times_ms.ravel()[index_min],
            "std_ms" : self.times_stdev.ravel()[index_min],
            "L0_min" : int(self.L0.ravel()[index_min]),
            "L1_min" : int(self.L1.ravel()[index_min]),
            "L2_min" : int(self.L2.ravel()[index_min]),
            "max_ms" : self.times_ms.ravel()[index_max],
            "std_ms_max" : self.times_stdev.ravel()[index_max],
            "L0_max" : int(self.L0.ravel()[index_max]),
            "L1_max" : int(self.L1.ravel()[index_max]),
            "L2_max" : int(self.L2.ravel()[index_max]),
            "times_ms" : list(self.times_ms.ravel()),
            "times_stdev" : list(self.times_stdev.ravel()),
            "local0" : [int(n) for n in self.local0],
            "local1" : [int(n) for n in self.local1],
            "local2" : [int(n) for n in self.local2]
        }
            
        # Report timings
        #self.report_timings(timing_data)
        return timing_data
        
class TimingPlotData:
    """Class to store the optimised timings for a number of algorithms"""
    def __init__(self):
        self.labels=[]
        self.colours=[]
        self.speedups=[]
        self.errors=[]
        
    def ingest(self, speedup, error, label, colour):
        """Ingest a single optimised result"""
        self.labels.append(label)
        self.colours.append(colour)
        self.speedups.append(speedup)
        self.errors.append(error)
        
    def num_items(self):
        """Get the number of results ingested"""
        return len(self.speedups)
        
class TimingResults:
    """Class to store and plot collections of timing results"""
    def __init__(self):
        """Constructor"""
        self.results = OrderedDict()
    
    def add_result(self, result, label, plot=False, benchmark=False):
        """Add a result to the dictionary, 
        label must be unique or else the 
        result will be overwritten.""" 
        if result is not None:
            if len(self.results)==0 or benchmark:
                self.benchmark_label = label
            self.results[label] = result
            
            if (plot):
                self.plot_result(label)
            
    
    def delete_result(self, label):
        """Remove a result from the dictionary"""
        if label in self.results:
            del self.results["label"]
    
    def plot_result(self, key):
        """Plot a single timing result"""
        result = LocalOpt(timings=self.results[key])
            
        # Make plots
        fig, axes = plt.subplots(1, 1, figsize=(6,6), sharex=True, sharey=True)
        ax=axes
                
        # Get data
        value = result.times_ms
            
        indices = np.where(~np.isnan(value))
        bad_indices=np.where(np.isnan(value))
    
        #value[bad_indices]=1.0
        value=np.log10(value)
        #value[bad_indices]=np.nan
                
        min_data = np.min(value[indices])
        max_data = np.max(value[indices])
                
        im = ax.imshow(value[...,0], vmin=min_data, vmax=max_data, origin="lower")
        #ax.contour(value, 20, origin="lower")
                
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Set labels on things
        ax.set_xticks(np.arange(0,result.local1.size,1))
        ax.set_yticks(np.arange(0,result.local0.size,1))
        ax.set_xticklabels([str(x) for x in result.local1])
        ax.set_yticklabels([str(x) for x in result.local0])
        ax.set_xlabel("Local size (dimension 1)")
        ax.set_ylabel("Local size (dimension 0)")
        ax.set_title("Time (ms)")

        # Put a color bar on the plot
        plt.colorbar(mappable=im, cax=cax)

        fig.tight_layout()
        plt.show()
        
    def plot_results(self, highlight_key=None):
        """Plot the collection of results, separate out the CPU and GPU results"""
        if len(self.results)>0:
            
            if highlight_key is None:
                highlight_key = self.benchmark_label
            
            # Sort by GPU results and CPU results
            
            # Make up timing results
            [fig, ax] = plt.subplots(2, 1, figsize=(6,6))
            
            t_bench = self.results[self.benchmark_label]["min_ms"]
            dt_bench = self.results[self.benchmark_label]["std_ms"]
            
            gpu_data = TimingPlotData()
            cpu_data = TimingPlotData()
            
            for key, result in self.results.items():
                
                # Get time
                t = result["min_ms"]
                # Get error in time
                dt = result["std_ms"]
                
                # Calculate speedup and associated error
                speedup  = t_bench / t
                err = (1/t)**2.0 * dt_bench**2.0
                err += (-t_bench/(t**2.0))**2.0 * dt**2.0
                err = math.sqrt(err)
                
                output=gpu_data
                if "CPU" in key:
                    output=cpu_data
                
                colour="Orange"       
                if highlight_key in key:
                    colour="Purple"
             
                output.ingest(speedup, err, key, colour)
            
            
            total_data = [*gpu_data.speedups, *cpu_data.speedups]
            
            for n, data in enumerate([cpu_data, gpu_data]):
                if data.num_items()>0:
                    
                    # Sort in descending order
                    sort_indices = (np.argsort(data.speedups))[::-1]
                    
                    ax[n].barh(np.array(data.labels)[sort_indices], 
                               np.array(data.speedups)[sort_indices], 
                               0.8, 
                               xerr=np.array(data.errors)[sort_indices], 
                               color=np.array(data.colours)[sort_indices])
                    ax[n].set_xlabel("Speedup, more is better")
                    ax[n].set_xlim((0,1.1*np.max(data.speedups)))
    
            fig.tight_layout()
            plt.show()
            
def plot_slices(images):
    """Function to plot a collection of images"""
    # Get the dimensions
    nslices, N0, N1 = images.shape
    
    # Animate the result
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    extent=[ -0.5, N1-0.5, -0.5, N0-0.5]
    img = ax.imshow(
        images[0,...], 
        extent=extent, 
        vmin=np.min(images), 
        vmax=np.max(images),
        cmap=plt.get_cmap("Greys_r")
    )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 0")
    ax.set_title("Images")

    def update(n=0):
        img.set_data(images[n,...])
        plt.show()
    
    # Run the interaction
    result = widgets.interact(
        update,
        n=(0, nslices-1, 1)
    )

def load_benchmark(filename):
    with open(filename, "r") as fd:
        result_json=" ".join(fd.readlines())
        results=json.loads(result_json)
        
    return results