import numpy as np
from matplotlib import pyplot as plt
# Import axes machinery
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        assert self.hasattr("C"), "Must run make_data() before check_data()."
        
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