'''
    Author: Antoine Falisse    

    This script uses OpenSimAD to generate a CasADi external function.
    
    Given an OpenSim model provided as an .osim file, this script generates a
    C++ file with a function F building the musculoskeletal model
    programmatically and running, among other, inverse dynamics. The C++ file
    is then compiled as an application which is run to generate the expression
    graph underlying the function F. From this expression graph, CasADi can
    generate C code containing the function F and its Jacobian in a format
    understandable by CasADi. This code is finally compiled as a dynamically
    linked library that can be imported when formulating trajectory
    optimization problems with CasADi.
    
    The function F takes as:
        - INPUTS: 
            - joint positions and velocities (intertwined)
            - joint accelerations
        - OUTPUTS:
            - joint torques
            - ground reaction forces
            - ground reaction moments
            - body origins
            
    You can adjust the script generateExternalFunction to modify the inputs or
    outputs.
            
    This script also saves a dictionnary F_map with the indices of the
    outputs of F. E.g., the left hip flexion index is given by 
    F_map['residuals']['hip_flexion_l'].
            
    See concrete example of how the function F can be used here:
        https://github.com/antoinefalisse/predsim_tutorial    
'''

import os
from utilities import generateExternalFunction

pathMain = os.getcwd()

# %% User inputs.
# Provide path to the directory where you want to save your results.
pathModelFolder = os.path.join(pathMain, 'examples')
# Provide path to OpenSim model.
modelName = 'Hamner_modified'
pathOpenSimModel = os.path.join(pathModelFolder, modelName + '.osim')
# Provide path to the InverseDynamics folder.
# To verify that what we did is correct, we compare torques returned by the
# external function given some input data to torques returned by OpenSim's ID
# tool given the same input data and the original .osim file. If the two sets
# of resulting torques differ, it means something went wrong when generating
# the external function.
pathID =  os.path.join(pathMain, 'InverseDynamics')

# %% Optional user inputs.
# Output file name (default is F).
outputFilename = modelName

# %% Generate external function.
generateExternalFunction(pathOpenSimModel, pathModelFolder, pathID,
                         outputFilename=outputFilename)

# %% Example (not recommended).
# You can also directly provide a cpp file and use the built-in utilities to
# build the corresponding dll. Note that with this approach, you will not get
# the F_map output.
# from utilities import buildExternalFunction
# nCoordinates = 31
# buildExternalFunction(outputFilename, pathModelFolder, 3*nCoordinates, 
#                       compiler=compiler)
