'''
    This script uses OpenSimAD to generate a CasADi external function. Given
    an OpenSim model provided as an .osim file, this script generates a C++
    file with a function F building the musculoskeletal model programmatically 
    and running inverse dynamics. The C++ file is then compiled as an .exe, 
    which when run generates the expression graph underlying F. From this
    expression graph, CasADi can generate C code containing the function F
    and its Jacobian in a format understandable by CasADi. This code is 
    finally compiled as a .dll that can be imported when formulating 
    trajectory optimization problems with CasADi.
    
    The function F takes as:
        - INPUTS: 
            - joint positions and velocities (intertwined)
            - joint accelerations
        - OUTPUTS:
            - joint torques
            - ground reaction forces
            - ground reaction moments
            - body origins
            
    This script also saves a dictionnary F_map with the indices of the
    outputs of F. E.g., the left hip flexion index is given by 
    F_map['residuals']['hip_flexion_l'].
            
    See concrete example of how the function F can be used here (TODO).        
    
    Author: Antoine Falisse
'''

import os
from utilities import generateExternalFunction

pathMain = os.getcwd()

# %% User inputs.
# Provide path to the directory where you want to save your results.
pathExample = os.path.join(pathMain, 'examples')
# Provide path to OpenSim model.
pathOpenSimModel = os.path.join(pathExample, 'Hamner_modified.osim')
# Provide path to the InverseDynamics folder.
# To verify that what we did is correct, we compare torques returned by the
# external function given some input data to torques returned by OpenSim's ID
# tool given the same input data and the original .osim file. If the two sets
# of resulting torques differ, it means something went wrong when generating
# the external function.
pathID =  os.path.join(pathMain, 'InverseDynamics')

# %% Optional user inputs.
# Output file name (default is F).
outputFilename = 'Hamner_modified_scaled'
# Compiler (default is "Visual Studio 15 2017 Win64").
compiler = "Visual Studio 15 2017 Win64"

# %% Generate external function.
generateExternalFunction(pathOpenSimModel, pathExample, pathID,
                         outputFilename=outputFilename, compiler=compiler)

# %% Example (not recommended).
# You can also directly provide a cpp file and use the built-in utilities to
# build the corresponding dll. Note that with this approach, you will not get
# the F_map output.
# from utilities import buildExternalFunction
# nCoordinates = 31
# buildExternalFunction(outputFilename, pathExample, 3*nCoordinates, 
#                       compiler=compiler)
