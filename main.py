import os

from utilities import generateExternalFunction


pathMain = os.getcwd()

pathExample = os.path.join(pathMain, 'examples')
pathOpenSimModel = os.path.join(pathExample, 'Hamner_modified.osim')
# pathOpenSimModel = os.path.join(pathExample, 'LaiArnold_modified.osim')
pathID =  os.path.join(pathMain, 'InverseDynamics')


# Specify the joint order you will use for the direct collocation problem.
# Find the joint names in the .osim file.
jointsOrder = ['ground_pelvis', 'hip_l', 'hip_r', 'knee_l', 'knee_r',
               'ankle_l', 'ankle_r', 'subtalar_l', 'subtalar_r', 'mtp_l',
               'mtp_r', 'back', 'acromial_l', 'acromial_r', 'elbow_l',
               'elbow_r', 'radioulnar_l', 'radioulnar_r', 'radius_hand_l',
               'radius_hand_r']
# Specify the corresponding coordinate order (for sanity check).
# Find the coordinate names in the .osim file in the join definitions.
coordinatesOrder = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 
    'pelvis_tz', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_l', 
    'knee_angle_r', 'ankle_angle_l', 'ankle_angle_r', 'subtalar_angle_l',
    'subtalar_angle_r', 'mtp_angle_l', 'mtp_angle_r', 'lumbar_extension', 
    'lumbar_bending', 'lumbar_rotation', 'arm_flex_l', 'arm_add_l', 
    'arm_rot_l', 'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_l', 
    'elbow_flex_r']


generateExternalFunction(pathOpenSimModel, pathExample, pathID,
                         jointsOrder, coordinatesOrder)
