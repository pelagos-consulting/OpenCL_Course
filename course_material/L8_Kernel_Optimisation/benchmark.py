import os
import numpy as np
import sys
import json
sys.path.insert(0, os.path.abspath("../include"))
from local_opt import LocalOpt
                
# Helper class to specify an experiment
class Exp:
    def __init__(
        self,
        cmds,
        local0=np.uint32(2**np.arange(0,10,1)),
        local1=np.uint32(2**np.arange(0,10,1)),
        local2=np.uint32(2**np.arange(0,1,1))):
        
        self.local0 = local0
        self.local1 = local1
        self.local2 = local2
        self.cmds = cmds.split()
        self.cmds[0] = os.path.join(os.getcwd(), self.cmds[0]) 

### Modify this section
                
output_file = "benchmark.json"
                
# Specify device types along with indices to try
dev_indices_cpu = [0]
dev_indices_gpu = [0,1]
dev_types={"cpu" : dev_indices_cpu, "gpu" : dev_indices_gpu}

# Specify programs to benchmark
experiments = {
    "Double precision" : "mat_mult_double.exe",
    "Single precision" : "mat_mult_float.exe",
    "Prefetch on A" : "mat_mult_prefetch.exe",
    "Local memory" : "mat_mult_local.exe",
    "Transpose B" : "mat_mult_BT.exe",
    "Transpose A" : "mat_mult_AT.exe",
    "Tile BT" : "mat_mult_tile_BT.exe",
    "Tile AT" : "mat_mult_tile_AT.exe",
    "Tile local BT" : "mat_mult_tile_local_BT.exe",
    "Tile vector BT" : "mat_mult_tile_vector_BT.exe",
    "Tile local vector BT" : "mat_mult_tile_local_vector_BT.exe",
    "CLBlast" : "mat_mult_clblast.exe"
}

# Make up the input specification
inputSpec={}
for exp, executable in experiments.items():
    for d, indices in dev_types.items():
        for i in indices:
            label=f"{exp} ({d.upper()})[{i}]"
            cmds=f"{executable} -{d} {i}"
            if exp=="CLBlast":
                inputSpec[label]=Exp(cmds, local0=2**(np.arange(0,1,1)), local1=2**(np.arange(0,1,1)))
            else:
                inputSpec[label]=Exp(cmds)
                
# Special cases here
inputSpec["CLBlast MD (GPU)"]=Exp("mat_mult_clblast_md.exe -gpu", local0=2**(np.arange(0,1,1)), local1=2**(np.arange(0,1,1)))
        
### Don't modify any more commands
       
results = {}
    
# Now process all commands
for label, spec in inputSpec.items():
    print(f"{label}, {spec.cmds}")
    # Make and run an optimisation experiment
    temp=LocalOpt(
        cmds=spec.cmds, 
        local0=spec.local0,
        local1=spec.local1,
        local2=spec.local2)
    
    if temp.has_data:
        # Results contains dictionary of timing results
        results[label]=temp.export_result()

# Put all results here
result_json=json.dumps(results)

with open(output_file, "w") as fd:
    fd.write(result_json)
                