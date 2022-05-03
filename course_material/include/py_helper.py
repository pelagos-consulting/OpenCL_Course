import numpy as np
from matplotlib import pyplot as plt
# Import axes machinery
from mpl_toolkits.axes_grid1 import make_axes_locatable
import subprocess
from collections.abc import Iterable

# Data type
dtype = np.float32

class MatMul:
    
    def __init__(self, NCOLS_A, NROWS_C, NCOLS_C, dtype):
        self.NCOLS_A = NCOLS_A
        self.NROWS_C = NROWS_C
        self.NCOLS_C = NCOLS_C
        self.dtype = dtype
        
    def make_data(self):
    
        # A is of size (NROWS_C, NCOLS_A)
        # B is of size (NCOLS_A, NCOLS_C)    
        # C is of size (NROWS_C, NCOLS_C)

        # Make up the arrays A, B, and C
        A = np.random.random(size = (self.NROWS_C, self.NCOLS_A)).astype(self.dtype)
        B = np.random.random(size = (self.NCOLS_A, self.NCOLS_C)).astype(self.dtype)

        # Make up the answer
        self.C = np.matmul(A, B, dtype = dtype)

        # Write out the arrays as binary files
        A.tofile("array_A.dat")
        B.tofile("array_B.dat")

    def check_data(self):
        # Make sure we have the solution
        assert hasattr(self, "C"), "Must run make_data() before check_data()."
        
        # Read in the output from OpenCL
        C_ocl = np.fromfile("array_C.dat", dtype=self.dtype).reshape((self.NROWS_C, self.NCOLS_C))

        # Make plots
        fig, axes = plt.subplots(3, 1, figsize=(6,8), sharex=True, sharey=True)

        # Data to plot
        data = [self.C, C_ocl, np.abs(self.C-C_ocl)]

        # Labels to plot
        labels = ["Numpy", "OpenCL", "Absolute residual"]

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
    
    def __init__(self, NROWS_F, NCOLS_F, dtype):
        self.NROWS_F = NROWS_F
        self.NCOLS_F = NCOLS_F
        self.dtype = dtype
        
    def make_data(self):
    
        # D is of size (NROWS_F, NCOLS_F)
        # E is of size (NCOLS_F, NCOLS_F)    
        # F is of size (NROWS_F, NCOLS_F)

        # Make up the arrays A, B, and C
        D = np.random.random(size = (self.NROWS_F, self.NCOLS_F)).astype(self.dtype)
        E = np.random.random(size = (self.NROWS_F, self.NCOLS_F)).astype(self.dtype)

        # Make up the answer using Hadamard multiplication
        self.F = D*E

        # Write out the arrays as binary files
        D.tofile("array_D.dat")
        E.tofile("array_E.dat")

    def check_data(self):
        # Make sure we have the solution
        assert hasattr(self, "F"), "Must run make_data() before check_data()."

        # Read in the output from OpenCL
        F_ocl = np.fromfile("array_F.dat", dtype=self.dtype).reshape((self.NROWS_F, self.NCOLS_F))

        # Make plots
        fig, axes = plt.subplots(3, 1, figsize=(6,8), sharex=True, sharey=True)

        # Data to plot
        data = [self.F, F_ocl, np.abs(self.F-F_ocl)]

        # Labels to plot
        labels = ["Numpy", "OpenCL", "Absolute residual"]

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
        
class LocalOpt2D:
    
    def __init__(self, ndim):
        self.ndim = ndim
        
    def make_data(self):
    
        # input_local.dat is of type np.uint32
        
        self.exp_vec0 = np.arange(0,10,1)
        self.exp_vec1 = np.arange(0,10,1)        
        
        self.local0 = np.uint32(2**self.exp_vec0)
        self.local1 = np.uint32(2**self.exp_vec1)
        
        # Turn local0 and local1 into a meshgrid of experiments
        V0, V1 = np.meshgrid(self.local0, self.local1, indexing="ij")
        
        self.nexperiments = self.local0.size*self.local1.size
        
        self.input_local = np.zeros((self.local0.size, self.local1.size, 2), dtype=np.uint32)
        self.input_local[:,:,0] = V0
        self.input_local[:,:,1] = V1
        
        # Write out to file
        self.input_local.tofile("input_local.dat")

    def run_problem(self, cmds):
        # Make sure we have the solution
        assert hasattr(self, "input_local"), "Must run make_data() before check_data()."

        # Add the --local-file flag
        if isinstance(cmds, Iterable) and not isinstance(cmds, str):
            temp_cmds = list(cmds) + ["--local_file"]
        else:
            temp_cmds=[cmds,"--local_file"]
        
        # Run the program 
        result = subprocess.run(temp_cmds)
        
        if (result.returncode==0):
        
            # Get the output data
            self.output_local = np.fromfile("output_local.dat", dtype=np.float64).reshape(
                self.local0.size, self.local1.size,2)

            # Make plots
            fig, axes = plt.subplots(1, 1, figsize=(6,6), sharex=True, sharey=True)

            # Data to plot
            #data = [self.output_local[:,:,0], self.output_local[:,:,1]]
            data = [self.output_local[:,:,0]]
            
            # Labels to plot
            labels = ["Time (ms)", "Error (ms)"]

            for n, value in enumerate(data):
                # Plot the graph
                #ax = axes[n]
                ax=axes
                
                indices = np.where(~np.isnan(value))
                bad_indices=np.where(np.isnan(value))
    
                value[bad_indices]=1.0
                value=np.log10(value)
                value[bad_indices]=np.nan
                
                min_data = np.min(value[indices])
                max_data = np.max(value[indices])
                
                im = ax.imshow(value, origin="lower")
                #ax.contour(value, 20, origin="lower")
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)

                # Set labels on things
                ax.set_xticks(self.exp_vec1)
                ax.set_yticks(self.exp_vec0)
                ax.set_xticklabels([str(x) for x in self.local1])
                ax.set_yticklabels([str(x) for x in self.local0])
                ax.set_xlabel("Local size (dimension 1)")
                ax.set_ylabel("Local size (dimension 0)")
                ax.set_title(labels[n])

                # Put a color bar on the plot
                plt.colorbar(mappable=im, cax=cax)

            fig.tight_layout()
            plt.show()