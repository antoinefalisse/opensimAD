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
            - (optional) other variables exported from the model
            
    See concrete example of how the function F can be used here (TODO).        
    
    Author: Antoine Falisse
'''

import os
from utilities import generateExternalFunction

pathMain = os.getcwd()

# %% User inputs.
# Paths.
# Provide path to the directory where you want to save your results.
pathExample = os.path.join(pathMain, 'examples')
# Provide path to OpenSim model.
pathOpenSimModel = os.path.join(pathExample, 'Hamner_modified_scaled.osim')
# Provide path to the InverseDynamics folder.
# To verify that what we did is correct, we compare torques returned by the
# external function given some input data to torques returned by OpenSim's ID
# tool given the same input data and the original .osim file. If the two sets
# of resulting torques differ, it means something went wrong when generating
# the external function.
pathID =  os.path.join(pathMain, 'InverseDynamics')
# Joints and coordinates.
# Provide the joints in the order you want to use when formulating your
# trajectory optimization problem. You should include all joints (ie, include
# also the weld joints). You can find the joint names in the .osim file.
jointsOrder = ['ground_pelvis', 'hip_l', 'hip_r', 'knee_l', 'knee_r',
               'ankle_l', 'ankle_r', 'subtalar_l', 'subtalar_r', 'mtp_l',
               'mtp_r', 'back', 'acromial_l', 'acromial_r', 'elbow_l',
               'elbow_r', 'radioulnar_l', 'radioulnar_r', 'radius_hand_l',
               'radius_hand_r']
# Provide the corresponding coordinates. Make sure the order match, eg. the
# 'ground_pelvis' joint has 6 coordinates, namely 'pelvis_tilt', 'pelvis_list',
# 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz' in this order.
coordinatesOrder = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 
    'pelvis_tz', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_l', 
    'knee_angle_r', 'ankle_angle_l', 'ankle_angle_r', 'subtalar_angle_l',
    'subtalar_angle_r', 'mtp_angle_l', 'mtp_angle_r', 'lumbar_extension', 
    'lumbar_bending', 'lumbar_rotation', 'arm_flex_l', 'arm_add_l', 
    'arm_rot_l', 'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_l', 
    'elbow_flex_r']

# %% Optional user inputs.
# Output file name (default is F).
outputFilename = 'Hamner_modified_scaled'
# Compiler (default is "Visual Studio 15 2017 Win64").
compiler = "Visual Studio 15 2017 Win64"

# By default, the external function returns the joint torques. However, you
# can also export other variables that you may want to use when formulating
# your problem. Here, we provide a few examples of variables we typically use
# in our problems. Note that the order matters, eg GRFs would be exported
# before 3D segment origins before GRMs.
# Export 2D segment origins.
# Leave empty or do not pass as argument to not export those variables.
export2DSegmentOrigins = ['calcn_r', 'calcn_l', 'femur_r', 'femur_l', 'hand_r',
                          'hand_l', 'tibia_r', 'tibia_l', 'toes_r', 'toes_l']
# Export GRFs.
# If True, right and left 3D GRFs (in this order) are exported. Set False or
# do not pass as argument to not export those variables.
exportGRFs = True
# # Export 3D segment origins.
# # Leave empty or do not pass as argument to not export those variables.
# export3DSegmentOrigins = ['pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r',
#                           'toes_r', 'femur_l', 'tibia_l', 'talus_l', 'calcn_l',
#                           'toes_l', 'torso', 'humerus_r', 'ulna_r', 'radius_r', 
#                           'hand_r', 'humerus_l', 'ulna_l', 'radius_l', 
#                           'hand_l']
# Export GRMs.
# If True, right and left 3D GRMs (in this order) are exported. Set False or
# do not pass as argument to not export those variables.
exportGRMs = True

# %% Generate external function.
generateExternalFunction(pathOpenSimModel, pathExample, pathID,
                         outputFilename=outputFilename,
                         compiler=compiler)
test = 0
