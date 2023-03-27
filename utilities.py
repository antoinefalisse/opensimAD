import os
import sys
import opensim
import numpy as np
import casadi as ca
import shutil
import importlib
import pandas as pd
import platform
import urllib.request
import zipfile

def generateExternalFunction(pathOpenSimModel, outputDir, pathID,
                             outputFilename='F'):
    
    # %% Paths.
    os.makedirs(outputDir, exist_ok=True)
    pathOutputFile = os.path.join(outputDir, outputFilename + ".cpp")
    pathOutputMap = os.path.join(outputDir, outputFilename + "_map.npy")
    
    # %% Generate external Function (.cpp file).
    opensim.Logger.setLevelString('error')
    model = opensim.Model(pathOpenSimModel)
    model.initSystem()
    bodySet = model.getBodySet()
    
    nBodies = 0
    for i in range(bodySet.getSize()):        
        c_body = bodySet.get(i)
        c_body_name = c_body.getName()  
        if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
            continue
        nBodies += 1
    
    jointSet = model.get_JointSet()
    nJoints = jointSet.getSize()
    geometrySet = model.get_ContactGeometrySet()
    forceSet = model.get_ForceSet()
    coordinateSet = model.getCoordinateSet()
    nCoordinates = coordinateSet.getSize()
    coordinates = []
    for coor in range(nCoordinates):
        coordinates.append(coordinateSet.get(coor).getName())
    sides = ['r', 'l']
    for side in sides:
        # We do not include the coordinates from the patellofemoral joints,
        # since they only influence muscle paths, which we approximate using
        # polynomials offline.
        if 'knee_angle_{}_beta'.format(side) in coordinates:
            nCoordinates -= 1
            nJoints -= 1
    
    nContacts = 0
    for i in range(forceSet.getSize()):        
        c_force_elt = forceSet.get(i)        
        if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":  
            nContacts += 1
    
    with open(pathOutputFile, "w") as f:
        
        # TODO: only include those that are necessary (model-specific).
        f.write('#include <OpenSim/Simulation/Model/Model.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/PinJoint.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/WeldJoint.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/PlanarJoint.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/Joint.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/SpatialTransform.h>\n')
        f.write('#include <OpenSim/Simulation/SimbodyEngine/CustomJoint.h>\n')
        f.write('#include <OpenSim/Common/LinearFunction.h>\n')
        f.write('#include <OpenSim/Common/PolynomialFunction.h>\n')
        f.write('#include <OpenSim/Common/MultiplierFunction.h>\n')
        f.write('#include <OpenSim/Common/Constant.h>\n')
        f.write('#include <OpenSim/Simulation/Model/SmoothSphereHalfSpaceForce.h>\n')
        f.write('#include <OpenSim/Simulation/SimulationUtilities.h>\n')
        f.write('#include "SimTKcommon/internal/recorder.h"\n\n')
        
        f.write('#include <iostream>\n')
        f.write('#include <iterator>\n')
        f.write('#include <random>\n')
        f.write('#include <cassert>\n')
        f.write('#include <algorithm>\n')
        f.write('#include <vector>\n')
        f.write('#include <fstream>\n\n')
        
        f.write('using namespace SimTK;\n')
        f.write('using namespace OpenSim;\n\n')
    
        f.write('constexpr int n_in = 2; \n')
        f.write('constexpr int n_out = 1; \n')
        
        f.write('constexpr int nCoordinates = %i; \n' % nCoordinates)
        f.write('constexpr int NX = nCoordinates*2; \n')
        f.write('constexpr int NU = nCoordinates; \n\n')
    
        f.write('template<typename T> \n')
        f.write('T value(const Recorder& e) { return e; }; \n')
        f.write('template<> \n')
        f.write('double value(const Recorder& e) { return e.getValue(); }; \n\n')
        
        f.write('template<typename T>\n')
        f.write('int F_generic(const T** arg, T** res) {\n\n')
        
        # Model
        f.write('\t// Definition of model.\n')
        f.write('\tOpenSim::Model* model;\n')
        f.write('\tmodel = new OpenSim::Model();\n\n')
        
        # Bodies
        f.write('\t// Definition of bodies.\n')
        for i in range(bodySet.getSize()):        
            c_body = bodySet.get(i)
            c_body_name = c_body.getName()            
            if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
                continue            
            c_body_mass = c_body.get_mass()
            c_body_mass_center = c_body.get_mass_center().to_numpy()
            c_body_inertia = c_body.get_inertia()
            c_body_inertia_vec3 = np.array([c_body_inertia.get(0), c_body_inertia.get(1), c_body_inertia.get(2)])        
            f.write('\tOpenSim::Body* %s;\n' % c_body_name)
            f.write('\t%s = new OpenSim::Body(\"%s\", %.20f, Vec3(%.20f, %.20f, %.20f), Inertia(%.20f, %.20f, %.20f, 0., 0., 0.));\n' % (c_body_name, c_body_name, c_body_mass, c_body_mass_center[0], c_body_mass_center[1], c_body_mass_center[2], c_body_inertia_vec3[0], c_body_inertia_vec3[1], c_body_inertia_vec3[2]))
            f.write('\tmodel->addBody(%s);\n' % (c_body_name))
            f.write('\n')
        
        # Joints
        f.write('\t// Definition of joints.\n')
        for i in range(jointSet.getSize()): 
            c_joint = jointSet.get(i)
            c_joint_type = c_joint.getConcreteClassName()
            
            c_joint_name = c_joint.getName()
            if (c_joint_name == 'patellofemoral_l' or 
                c_joint_name == 'patellofemoral_r'):
                continue
            
            parent_frame = c_joint.get_frames(0)
            parent_frame_name = parent_frame.getParentFrame().getName()
            parent_frame_trans = parent_frame.get_translation().to_numpy()
            parent_frame_or = parent_frame.get_orientation().to_numpy()
            
            child_frame = c_joint.get_frames(1)
            child_frame_name = child_frame.getParentFrame().getName()
            child_frame_trans = child_frame.get_translation().to_numpy()
            child_frame_or = child_frame.get_orientation().to_numpy()
            
            # Custom joints
            if c_joint_type == "CustomJoint":

                f.write('\tSpatialTransform st_%s;\n' % c_joint.getName())

                cObj = opensim.CustomJoint.safeDownCast(c_joint)
                spatialtransform = cObj.get_SpatialTransform()

                for iCoord in range(6):
                    if iCoord == 0:
                        dofSel = spatialtransform.get_rotation1()
                    elif iCoord == 1:
                        dofSel = spatialtransform.get_rotation2()
                    elif iCoord == 2:
                        dofSel = spatialtransform.get_rotation3()
                    elif iCoord == 3:
                        dofSel = spatialtransform.get_translation1()
                    elif iCoord == 4:
                        dofSel = spatialtransform.get_translation2()
                    elif iCoord == 5:
                        dofSel = spatialtransform.get_translation3()
                    coord = iCoord

                    # Transform axis.
                    dofSel_axis = dofSel.get_axis().to_numpy()
                    dofSel_f = dofSel.get_function()
                    if dofSel_f.getConcreteClassName() == 'LinearFunction':
                        dofSel_f_obj = opensim.LinearFunction.safeDownCast(dofSel_f)
                        dofSel_f_slope = dofSel_f_obj.getSlope()
                        dofSel_f_intercept = dofSel_f_obj.getIntercept()
                        #c_coord = c_joint.get_coordinates(coord)
                        c_coord_name = dofSel.get_coordinates(0)
                        f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (
                        c_joint.getName(), coord, c_coord_name))
                        f.write('\tst_%s[%i].setFunction(new LinearFunction(%.4f, %.4f));\n' % (
                        c_joint.getName(), coord, dofSel_f_slope, dofSel_f_intercept))
                    elif dofSel_f.getConcreteClassName() == 'PolynomialFunction':
                        f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (
                        c_joint.getName(), coord, c_coord_name))
                        dofSel_f_obj = opensim.PolynomialFunction.safeDownCast(dofSel_f)
                        dofSel_f_coeffs = dofSel_f_obj.getCoefficients().to_numpy()
                        c_nCoeffs = dofSel_f_coeffs.shape[0]
                        if c_nCoeffs == 2:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (
                            c_joint.getName(), coord, c_nCoeffs, dofSel_f_coeffs[0], dofSel_f_coeffs[1]))
                        elif c_nCoeffs == 3:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (
                            c_joint.getName(), coord, c_nCoeffs, dofSel_f_coeffs[0], dofSel_f_coeffs[1], dofSel_f_coeffs[2]))
                        elif c_nCoeffs == 4:
                            f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (
                            c_joint.getName(), coord, c_nCoeffs, dofSel_f_coeffs[0], dofSel_f_coeffs[1], dofSel_f_coeffs[2],
                            dofSel_f_coeffs[3]))
                        elif c_nCoeffs == 5:
                            f.write(
                                '\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (
                                c_joint.getName(), coord, c_nCoeffs, dofSel_f_coeffs[0], dofSel_f_coeffs[1], dofSel_f_coeffs[2],
                                dofSel_f_coeffs[3], dofSel_f_coeffs[4]))
                        elif c_nCoeffs == 7:
                            f.write(
                                '\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (
                                c_joint.getName(), coord, c_nCoeffs, dofSel_f_coeffs[0], dofSel_f_coeffs[1], dofSel_f_coeffs[2],
                                dofSel_f_coeffs[3], dofSel_f_coeffs[4], dofSel_f_coeffs[5], dofSel_f_coeffs[6]))
                        else:
                            raise ValueError("TODO")
                        f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                        f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (
                        c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                        f.write('\tst_%s[%i].setFunction(new PolynomialFunction(st_%s_%i_coeffs_vec));\n' % (
                        c_joint.getName(), coord, c_joint.getName(), coord))
                    elif dofSel_f.getConcreteClassName() == 'MultiplierFunction':
                        dofSel_f_obj = opensim.MultiplierFunction.safeDownCast(dofSel_f)
                        dofSel_f_obj_scale = dofSel_f_obj.getScale()
                        dofSel_f_obj_f = dofSel_f_obj.getFunction()
                        dofSel_f_obj_f_name = dofSel_f_obj_f.getConcreteClassName()
                        if dofSel_f_obj_f_name == 'Constant':
                            dofSel_f_obj_f_obj = opensim.Constant.safeDownCast(dofSel_f_obj_f)
                            dofSel_f_obj_f_obj_value = dofSel_f_obj_f_obj.getValue()
                            f.write('\tst_%s[%i].setFunction(new MultiplierFunction(new Constant(%.20f), %.20f));\n' % (
                            c_joint.getName(), coord, dofSel_f_obj_f_obj_value, dofSel_f_obj_scale))
                        elif dofSel_f_obj_f_name == 'PolynomialFunction':
                            f.write('\tst_%s[%i].setCoordinateNames(OpenSim::Array<std::string>(\"%s\", 1, 1));\n' % (
                            c_joint.getName(), coord, c_coord_name))
                            dofSel_f_obj_f_obj = opensim.PolynomialFunction.safeDownCast(dofSel_f_obj_f)
                            dofSel_f_obj_f_coeffs = dofSel_f_obj_f_obj.getCoefficients().to_numpy()
                            c_nCoeffs = dofSel_f_obj_f_coeffs.shape[0]
                            if c_nCoeffs == 2:
                                f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f}; \n' % (
                                c_joint.getName(), coord, c_nCoeffs, dofSel_f_obj_f_coeffs[0], dofSel_f_obj_f_coeffs[1]))
                            elif c_nCoeffs == 3:
                                f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f}; \n' % (
                                c_joint.getName(), coord, c_nCoeffs, dofSel_f_obj_f_coeffs[0], dofSel_f_obj_f_coeffs[1],
                                dofSel_f_obj_f_coeffs[2]))
                            elif c_nCoeffs == 4:
                                f.write('\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f}; \n' % (
                                c_joint.getName(), coord, c_nCoeffs, dofSel_f_obj_f_coeffs[0], dofSel_f_obj_f_coeffs[1],
                                dofSel_f_obj_f_coeffs[2], dofSel_f_obj_f_coeffs[3]))
                            elif c_nCoeffs == 5:
                                f.write(
                                    '\tosim_double_adouble st_%s_%i_coeffs[%i] = {%.20f, %.20f, %.20f, %.20f, %.20f}; \n' % (
                                    c_joint.getName(), coord, c_nCoeffs, dofSel_f_obj_f_coeffs[0], dofSel_f_obj_f_coeffs[1],
                                    dofSel_f_obj_f_coeffs[2], dofSel_f_obj_f_coeffs[3], dofSel_f_obj_f_coeffs[4]))
                            else:
                                raise ValueError("TODO")
                            f.write('\tVector st_%s_%i_coeffs_vec(%i); \n' % (c_joint.getName(), coord, c_nCoeffs))
                            f.write('\tfor (int i = 0; i < %i; ++i) st_%s_%i_coeffs_vec[i] = st_%s_%i_coeffs[i]; \n' % (
                            c_nCoeffs, c_joint.getName(), coord, c_joint.getName(), coord))
                            f.write(
                                '\tst_%s[%i].setFunction(new MultiplierFunction(new PolynomialFunction(st_%s_%i_coeffs_vec), %.20f));\n' % (
                                c_joint.getName(), coord, c_joint.getName(), coord, dofSel_f_obj_scale))
                        else:
                            raise ValueError("Not supported")
                    elif dofSel_f.getConcreteClassName() == 'Constant':
                        dofSel_f_obj = opensim.Constant.safeDownCast(dofSel_f)
                        dofSel_f_obj_value = dofSel_f_obj.getValue()
                        f.write('\tst_%s[%i].setFunction(new Constant(%.20f));\n' % (
                        c_joint.getName(), coord, dofSel_f_obj_value))
                    else:
                        raise ValueError(dofSel_f.getConcreteClassName() +" Not supported")
                    f.write('\tst_%s[%i].setAxis(Vec3(%.20f, %.20f, %.20f));\n' % (
                    c_joint.getName(), coord, dofSel_axis[0], dofSel_axis[1], dofSel_axis[2]))
                
                
                # Joint.
                f.write('\tOpenSim::%s* %s;\n' % (c_joint_type, c_joint.getName()))
                if parent_frame_name == "ground":
                    f.write('\t%s = new OpenSim::%s(\"%s\", model->getGround(), Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), st_%s);\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2], c_joint.getName()))     
                else:
                    f.write('\t%s = new OpenSim::%s(\"%s\", *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), st_%s);\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_name, parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2], c_joint.getName()))
                
            elif c_joint_type == 'PinJoint' or c_joint_type == 'WeldJoint' or c_joint_type == 'PlanarJoint':
                f.write('\tOpenSim::%s* %s;\n' % (c_joint_type, c_joint.getName()))
                if parent_frame_name == "ground":
                    f.write('\t%s = new OpenSim::%s(\"%s\", model->getGround(), Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2]))     
                else:
                    f.write('\t%s = new OpenSim::%s(\"%s\", *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f), *%s, Vec3(%.20f, %.20f, %.20f), Vec3(%.20f, %.20f, %.20f));\n' % (c_joint.getName(), c_joint_type, c_joint.getName(), parent_frame_name, parent_frame_trans[0], parent_frame_trans[1], parent_frame_trans[2], parent_frame_or[0], parent_frame_or[1], parent_frame_or[2], child_frame_name, child_frame_trans[0], child_frame_trans[1], child_frame_trans[2], child_frame_or[0], child_frame_or[1], child_frame_or[2])) 
            else:
                raise ValueError("TODO: joint type not yet supported")
            f.write('\tmodel->addJoint(%s);\n' % (c_joint.getName()))
            f.write('\n')  
                
        # Contacts
        f.write('\t// Definition of contacts.\n')   
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)
            
            rightFootContact = False
            leftFootContact = False
            
            if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":            
                c_force_elt_obj =  opensim.SmoothSphereHalfSpaceForce.safeDownCast(c_force_elt) 	
                
                socket0Name = c_force_elt.getSocketNames()[0]
                socket0 = c_force_elt.getSocket(socket0Name)
                socket0_obj = socket0.getConnecteeAsObject()
                socket0_objName = socket0_obj.getName()            
                geo0 = geometrySet.get(socket0_objName)
                geo0_loc = geo0.get_location().to_numpy()
                geo0_or = geo0.get_orientation().to_numpy()
                geo0_frameName = geo0.getFrame().getName()
                
                socket1Name = c_force_elt.getSocketNames()[1]
                socket1 = c_force_elt.getSocket(socket1Name)
                socket1_obj = socket1.getConnecteeAsObject()
                socket1_objName = socket1_obj.getName()            
                geo1 = geometrySet.get(socket1_objName)
                geo1_loc = geo1.get_location().to_numpy()
                # geo1_or = geo1.get_orientation().to_numpy()
                geo1_frameName = geo1.getFrame().getName()
                obj = opensim.ContactSphere.safeDownCast(geo1) 	
                geo1_radius = obj.getRadius()            
                
                f.write('\tOpenSim::%s* %s;\n' % (c_force_elt.getConcreteClassName(), c_force_elt.getName()))
                if geo0_frameName == "ground":
                    f.write('\t%s = new %s(\"%s\", *%s, model->getGround());\n' % (c_force_elt.getName(), c_force_elt.getConcreteClassName(), c_force_elt.getName(), geo1_frameName))
                else:
                    f.write('\t%s = new %s(\"%s\", *%s, *%s);\n' % (c_force_elt.getName(), c_force_elt.getConcreteClassName(), c_force_elt.getName(), geo1_frameName, geo0_frameName))
                    
                f.write('\tVec3 %s_location(%.20f, %.20f, %.20f);\n' % (c_force_elt.getName(), geo1_loc[0], geo1_loc[1], geo1_loc[2]))
                f.write('\t%s->set_contact_sphere_location(%s_location);\n' % (c_force_elt.getName(), c_force_elt.getName()))
                f.write('\tdouble %s_radius = (%.20f);\n' % (c_force_elt.getName(), geo1_radius))
                f.write('\t%s->set_contact_sphere_radius(%s_radius );\n' % (c_force_elt.getName(), c_force_elt.getName()))
                f.write('\t%s->set_contact_half_space_location(Vec3(%.20f, %.20f, %.20f));\n' % (c_force_elt.getName(), geo0_loc[0], geo0_loc[1], geo0_loc[2]))
                f.write('\t%s->set_contact_half_space_orientation(Vec3(%.20f, %.20f, %.20f));\n' % (c_force_elt.getName(), geo0_or[0], geo0_or[1], geo0_or[2]))
                
                f.write('\t%s->set_stiffness(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_stiffness()))
                f.write('\t%s->set_dissipation(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_dissipation()))
                f.write('\t%s->set_static_friction(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_static_friction()))
                f.write('\t%s->set_dynamic_friction(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_dynamic_friction()))
                f.write('\t%s->set_viscous_friction(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_viscous_friction()))
                f.write('\t%s->set_transition_velocity(%.20f);\n' % (c_force_elt.getName(), c_force_elt_obj.get_transition_velocity()))
                
                f.write('\t%s->connectSocket_sphere_frame(*%s);\n' % (c_force_elt.getName(), geo1_frameName))
                if geo0_frameName == "ground":
                    f.write('\t%s->connectSocket_half_space_frame(model->getGround());\n' % (c_force_elt.getName()))                
                else:
                    f.write('\t%s->connectSocket_half_space_frame(*%s);\n' % (c_force_elt.getName(), geo0_frameName))
                f.write('\tmodel->addComponent(%s);\n' % (c_force_elt.getName()))
                f.write('\n')
                
                # Check if there are right and left foot contacts
                if not rightFootContact and c_force_elt.getName()[-2:] == '_r':
                    rightFootContact = True
                if not leftFootContact and c_force_elt.getName()[-2:] == '_l':
                    leftFootContact = True
                    
                
        f.write('\t// Initialize system.\n')
        f.write('\tSimTK::State* state;\n')
        f.write('\tstate = new State(model->initSystem());\n\n')
    
        f.write('\t// Read inputs.\n')
        f.write('\tstd::vector<T> x(arg[0], arg[0] + NX);\n')
        f.write('\tstd::vector<T> u(arg[1], arg[1] + NU);\n\n')
        
        f.write('\t// States and controls.\n')
        f.write('\tT ua[NU];\n')
        f.write('\tVector QsUs(NX);\n')
        f.write('\t/// States\n')
        f.write('\tfor (int i = 0; i < NX; ++i) QsUs[i] = x[i];\n') 	
        f.write('\t/// Controls\n')
        f.write('\t/// OpenSim and Simbody have different state orders.\n')
        f.write('\tauto indicesOSInSimbody = getIndicesOpenSimInSimbody(*model);\n')
        f.write('\tfor (int i = 0; i < NU; ++i) ua[i] = u[indicesOSInSimbody[i]];\n\n')
    
        f.write('\t// Set state variables and realize.\n')
        f.write('\tmodel->setStateVariableValues(*state, QsUs);\n')
        f.write('\tmodel->realizeVelocity(*state);\n\n')
        
        f.write('\t// Compute residual forces.\n')
        f.write('\t/// Set appliedMobilityForces (# mobilities).\n')
        f.write('\tVector appliedMobilityForces(nCoordinates);\n')
        f.write('\tappliedMobilityForces.setToZero();\n')
        f.write('\t/// Set appliedBodyForces (# bodies + ground).\n')
        f.write('\tVector_<SpatialVec> appliedBodyForces;\n')
        f.write('\tint nbodies = model->getBodySet().getSize() + 1;\n')
        f.write('\tappliedBodyForces.resize(nbodies);\n')
        f.write('\tappliedBodyForces.setToZero();\n')
        f.write('\t/// Set gravity.\n')
        f.write('\tVec3 gravity(0);\n')
        f.write('\tgravity[1] = %.20f;\n' % model.get_gravity()[1])
        f.write('\t/// Add weights to appliedBodyForces.\n')
        f.write('\tfor (int i = 0; i < model->getBodySet().getSize(); ++i) {\n')
        f.write('\t\tmodel->getMatterSubsystem().addInStationForce(*state,\n')
        f.write('\t\tmodel->getBodySet().get(i).getMobilizedBodyIndex(),\n')
        f.write('\t\tmodel->getBodySet().get(i).getMassCenter(),\n')
        f.write('\t\tmodel->getBodySet().get(i).getMass()*gravity, appliedBodyForces);\n')
        f.write('\t}\n')    
        f.write('\t/// Add contact forces to appliedBodyForces.\n')
        
        count = 0
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)     
            
            if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":
                c_force_elt_name = c_force_elt.getName()    
                
                f.write('\tArray<osim_double_adouble> Force_%s = %s->getRecordValues(*state);\n' % (str(count), c_force_elt_name))
                f.write('\tSpatialVec GRF_%s;\n' % (str(count)))           
                
                f.write('\tGRF_%s[0] = Vec3(Force_%s[3], Force_%s[4], Force_%s[5]);\n' % (str(count), str(count), str(count), str(count)))
                f.write('\tGRF_%s[1] = Vec3(Force_%s[0], Force_%s[1], Force_%s[2]);\n' % (str(count), str(count), str(count), str(count)))
                
                socket1Name = c_force_elt.getSocketNames()[1]
                socket1 = c_force_elt.getSocket(socket1Name)
                socket1_obj = socket1.getConnecteeAsObject()
                socket1_objName = socket1_obj.getName()            
                geo1 = geometrySet.get(socket1_objName)
                geo1_frameName = geo1.getFrame().getName()
                
                f.write('\tint c_idx_%s = model->getBodySet().get("%s").getMobilizedBodyIndex();\n' % (str(count), geo1_frameName))            
                f.write('\tappliedBodyForces[c_idx_%s] += GRF_%s;\n' % (str(count), str(count)))
                count += 1
                f.write('\n')
                
        f.write('\t/// knownUdot.\n')
        f.write('\tVector knownUdot(nCoordinates);\n')
        f.write('\tknownUdot.setToZero();\n')
        f.write('\tfor (int i = 0; i < nCoordinates; ++i) knownUdot[i] = ua[i];\n')
        f.write('\t/// Calculate residual forces.\n')
        f.write('\tVector residualMobilityForces(nCoordinates);\n')
        f.write('\tresidualMobilityForces.setToZero();\n')
        f.write('\tmodel->getMatterSubsystem().calcResidualForceIgnoringConstraints(*state,\n')
        f.write('\t\t\tappliedMobilityForces, appliedBodyForces, knownUdot, residualMobilityForces);\n\n')
        
        # Get body origins.
        f.write('\t/// Body origins.\n')
        for i in range(bodySet.getSize()):        
            c_body = bodySet.get(i)
            c_body_name = c_body.getName()            
            if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
                continue            
            f.write('\tVec3 %s_or = %s->getPositionInGround(*state);\n' % (c_body_name, c_body_name))
        f.write('\n')
            
        # Get GRFs.
        f.write('\t/// Ground reaction forces.\n')
        if rightFootContact:
            f.write('\tVec3 GRF_r(0);\n')
        if leftFootContact:
            f.write('\tVec3 GRF_l(0);\n')
        count = 0
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)  
            if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":
                c_force_elt_name = c_force_elt.getName() 
                if c_force_elt_name[-2:] == "_r":
                    f.write('\tGRF_r += GRF_%s[1];\n'  % (str(count)))
                elif c_force_elt_name[-2:] == "_l":
                    f.write('\tGRF_l += GRF_%s[1];\n'  % (str(count)))
                else:
                    raise ValueError("Cannot identify contact side")
                count += 1
        f.write('\n')
            
        # Get GRMs.
        f.write('\t/// Ground reaction moments.\n')
        if rightFootContact:
            f.write('\tVec3 GRM_r(0);\n')
        if leftFootContact:
            f.write('\tVec3 GRM_l(0);\n')
        f.write('\tVec3 normal(0, 1, 0);\n\n')
        count = 0
        geo1_frameNames = []
        for i in range(forceSet.getSize()):        
            c_force_elt = forceSet.get(i)  
            if c_force_elt.getConcreteClassName() == "SmoothSphereHalfSpaceForce":
                c_force_elt_name = c_force_elt.getName() 
                socket1Name = c_force_elt.getSocketNames()[1]
                socket1 = c_force_elt.getSocket(socket1Name)
                socket1_obj = socket1.getConnecteeAsObject()
                socket1_objName = socket1_obj.getName()            
                geo1 = geometrySet.get(socket1_objName)
                geo1_frameName = geo1.getFrame().getName() 
                
                if not geo1_frameName in geo1_frameNames:
                    f.write('\tSimTK::Transform TR_GB_%s = %s->getMobilizedBody().getBodyTransform(*state);\n' % (geo1_frameName, geo1_frameName))    
                    geo1_frameNames.append(geo1_frameName)
                    
                f.write('\tVec3 %s_location_G = %s->findStationLocationInGround(*state, %s_location);\n' % (c_force_elt_name, geo1_frameName, c_force_elt_name))                
                f.write('\tVec3 %s_locationCP_G = %s_location_G - %s_radius * normal;\n' % (c_force_elt_name, c_force_elt_name, c_force_elt_name))
                f.write('\tVec3 locationCP_G_adj_%i = %s_locationCP_G - 0.5*%s_locationCP_G[1] * normal;\n' % (count, c_force_elt_name, c_force_elt_name))
                f.write('\tVec3 %s_locationCP_B = model->getGround().findStationLocationInAnotherFrame(*state, locationCP_G_adj_%i, *%s);\n' % (c_force_elt_name, count, geo1_frameName))
                f.write('\tVec3 GRM_%i = (TR_GB_%s*%s_locationCP_B) %% GRF_%s[1];\n' % (count, geo1_frameName, c_force_elt_name, str(count)))
                
                if c_force_elt_name[-2:] == "_r":
                    f.write('\tGRM_r += GRM_%i;\n'  % (count))   
                elif c_force_elt_name[-2:] == "_l": 
                    f.write('\tGRM_l += GRM_%i;\n'  % (count))   
                else:
                    raise ValueError("Cannot identify contact side")
                f.write('\n')                   
                count += 1
        
        # Save dict pointing to which elements are returned by F and in which
        # order, such as to facilitate using F when formulating problem.
        F_map = {}
        
        f.write('\t/// Outputs.\n')        
        # Export residuals (joint torques).
        f.write('\t/// Residual forces (OpenSim and Simbody have different state orders).\n')
        f.write('\tauto indicesSimbodyInOS = getIndicesSimbodyInOpenSim(*model);\n')
        f.write('\tfor (int i = 0; i < NU; ++i) res[0][i] =\n')
        f.write('\t\t\tvalue<T>(residualMobilityForces[indicesSimbodyInOS[i]]);\n')
        F_map['residuals'] = {}
        count = 0
        for coordinate in coordinates:
            if 'beta' in coordinate:
                continue
            F_map['residuals'][coordinate] = count 
            count += 1
        count_acc = nCoordinates
        
        # Export GRFs
        f.write('\t/// Ground reaction forces.\n')
        F_map['GRFs'] = {}
        if rightFootContact:
            f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(GRF_r[i]);\n' % (count_acc))
            F_map['GRFs']['right'] = range(count_acc, count_acc+3)
            count_acc += 3
        if leftFootContact:
            f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(GRF_l[i]);\n' % (count_acc))
            F_map['GRFs']['left'] = range(count_acc, count_acc+3)
            count_acc += 3
        
        # Export GRMs
        f.write('\t/// Ground reaction moments.\n')
        F_map['GRMs'] = {}
        if rightFootContact:
            f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(GRM_r[i]);\n' % (count_acc))
            F_map['GRMs']['right'] = range(count_acc, count_acc+3)
            count_acc += 3
        if leftFootContact:
            f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(GRM_l[i]);\n' % (count_acc))
            F_map['GRMs']['left'] = range(count_acc, count_acc+3)
            count_acc += 3
        
        # Export body origins.        
        f.write('\t/// Body origins.\n')
        F_map['body_origins'] = {}
        count = 0
        for i in range(bodySet.getSize()):        
            c_body = bodySet.get(i)
            c_body_name = c_body.getName()
            if (c_body_name == 'patella_l' or c_body_name == 'patella_r'):
                continue
            f.write('\tfor (int i = 0; i < 3; ++i) res[0][i + %i] = value<T>(%s_or[i]);\n' % (count_acc+count*3, c_body_name))
            F_map['body_origins'][c_body_name] = range(count_acc+count*3, count_acc+count*3+3)
            count += 1
        count_acc += 3*count
            
        f.write('\n')
        f.write('\treturn 0;\n')
        f.write('}\n\n')
        
        # Residuals (joint torques), 3D GRFs, GRMs, and body origins.
        nOutputs = nCoordinates + 3*nBodies
        if rightFootContact:
            nOutputs += 2*3
        if leftFootContact:
            nOutputs += 2*3
        
        f.write('constexpr int NR = %i; \n\n' % (nOutputs))
        
        f.write('int main() {\n')
        f.write('\tRecorder x[NX];\n')
        f.write('\tRecorder u[NU];\n')
        f.write('\tRecorder tau[NR];\n')
        f.write('\tfor (int i = 0; i < NX; ++i) x[i] <<= 0;\n')
        f.write('\tfor (int i = 0; i < NU; ++i) u[i] <<= 0;\n')
        f.write('\tconst Recorder* Recorder_arg[n_in] = { x,u };\n')
        f.write('\tRecorder* Recorder_res[n_out] = { tau };\n')
        f.write('\tF_generic<Recorder>(Recorder_arg, Recorder_res);\n')
        f.write('\tdouble res[NR];\n')
        f.write('\tfor (int i = 0; i < NR; ++i) Recorder_res[0][i] >>= res[i];\n')
        f.write('\tRecorder::stop_recording();\n')
        f.write('\treturn 0;\n')
        f.write('}\n')
        
        # Save dict
        np.save(pathOutputMap, F_map)
        
    # %% Build external Function.
    buildExternalFunction(outputFilename, outputDir, 3*nCoordinates)
        
    # %% Torque verification test.
    # Delete previous saved dummy motion if needed.
    if os.path.exists(os.path.join(pathID, "dummyData.sto")):
        os.remove(os.path.join(pathID, "dummyData.sto"))

    # Create a dummy motion for ID.
    nCoordinatesAll = coordinateSet.getSize()
    dummyDataa = np.zeros((10, nCoordinatesAll + 1))
    for coor in range(nCoordinatesAll):
        dummyDataa[:, coor + 1] = np.random.rand()*0.05
    dummyDataa[:, 0] = np.linspace(0.01, 0.1, 10)
    labelsDummy = []
    labelsDummy.append("time")
    for coor in range(nCoordinatesAll):
        labelsDummy.append(coordinateSet.get(coor).getName())
    numpy2storage(labelsDummy, dummyDataa, os.path.join(pathID,
                                                        "dummyData.sto"))

    # Solve inverse dynamics.
    pathGenericIDSetupFile = os.path.join(pathID, "SetupID.xml")
    idTool = opensim.InverseDynamicsTool(pathGenericIDSetupFile)
    idTool.setName("ID_withOsimAndIDTool")
    idTool.setModelFileName(pathOpenSimModel)
    idTool.setResultsDir(outputDir)
    idTool.setCoordinatesFileName(os.path.join(pathID, "dummyData.sto"))
    idTool.setOutputGenForceFileName("ID_withOsimAndIDTool.sto")       
    pathSetupID = os.path.join(outputDir, "SetupID.xml")
    idTool.printToXML(pathSetupID)
    idTool.run()
    
    # Extract torques from .osim + ID tool.    
    headers = []    
    for coord in range(nCoordinatesAll):                
        if (coordinateSet.get(coord).getName() == "pelvis_tx" or 
            coordinateSet.get(coord).getName() == "pelvis_ty" or 
            coordinateSet.get(coord).getName() == "pelvis_tz" or
            coordinateSet.get(coord).getName() == "knee_angle_r_beta" or 
            coordinateSet.get(coord).getName() == "knee_angle_l_beta"):
            suffix_header = "_force"
        else:
            suffix_header = "_moment"
        headers.append(coordinateSet.get(coord).getName() + suffix_header)
        
    from utilities import storage2df    
    ID_osim_df = storage2df(os.path.join(outputDir,
                                  "ID_withOsimAndIDTool.sto"), headers)
    ID_osim = np.zeros((nCoordinates))
    count = 0
    for coordinate in coordinates:
        if (coordinate == "pelvis_tx" or 
            coordinate == "pelvis_ty" or 
            coordinate == "pelvis_tz"):
            suffix_header = "_force"
        else:
            suffix_header = "_moment"
        if 'beta' in coordinate:
            continue                
        ID_osim[count] = ID_osim_df.iloc[0][coordinate + suffix_header]
        count += 1
    
    # Extract torques from external function.
    os_system = platform.system()
    if os_system == 'Windows':
        F = ca.external('F', os.path.join(outputDir, 
                                          outputFilename + '.dll'))
    elif os_system == 'Linux':
        F = ca.external('F', os.path.join(outputDir, 
                                          outputFilename + '.so'))
    elif os_system == 'Darwin':
        F = ca.external('F', os.path.join(outputDir, 
                                          outputFilename + '.dylib')) 
    DefaultPos = storage2df(os.path.join(pathID,
                                         "dummyData.sto"), coordinates)
    vecInput = np.zeros((nCoordinates * 3, 1))    
    coordinates_sel = []
    for coord in coordinates:
        if 'beta' in coord:
            continue
        coordinates_sel.append(coord)        
    idxCoord4F = [coordinates_sel.index(coord) 
                  for coord in list(F_map['residuals'].keys())]
    for c, coor in enumerate(coordinates_sel):        
        vecInput[idxCoord4F[c] * 2] = DefaultPos.iloc[0][coor]
    ID_F = (F(vecInput)).full().flatten()[:nCoordinates]
    
    # Verify torques from external match torques from .osim + ID tool.
    assert(np.max(np.abs(ID_osim - ID_F)) < 1e-6), (
        "Torque verification test failed")
    print('Torque verification test passed')

# %% Generate c-code with external function (and its Jacobian).
def generateF(dim):
    import foo
    importlib.reload(foo)
    cg = ca.CodeGenerator('foo_jac')
    arg = ca.SX.sym('arg', dim)
    y,_,_ = foo.foo(arg)
    F = ca.Function('F',[arg],[y])
    cg.add(F)
    cg.add(F.jacobian())
    cg.generate()

# %% Build/compile external function.
def buildExternalFunction(filename, CPP_DIR, nInputs):       
    
    # %% Part 1: build expression graph (i.e., generate foo.py).
    pathMain = os.getcwd()
    pathBuildExpressionGraph = os.path.join(pathMain, 'buildExpressionGraph')
    pathBuild = os.path.join(pathMain, 'build-ExpressionGraph' + filename)
    os.makedirs(pathBuild, exist_ok=True)
    OpenSimAD_DIR = os.path.join(pathMain, 'opensimAD-install')
    os.makedirs(OpenSimAD_DIR, exist_ok=True)
    os_system = platform.system()
    
    if os_system == 'Windows':
        pathBuildExpressionGraphOS = os.path.join(pathBuildExpressionGraph, 'windows') 
        OpenSimADOS_DIR = os.path.join(OpenSimAD_DIR, 'windows')        
        BIN_DIR = os.path.join(OpenSimADOS_DIR, 'bin')
        SDK_DIR = os.path.join(OpenSimADOS_DIR, 'sdk')
        # Download libraries if not existing locally.
        if not os.path.exists(BIN_DIR):
            url = 'https://sourceforge.net/projects/opensimad/files/windows.zip'
            zipfilename = 'windows.zip'                
            download_file(url, zipfilename)
            with zipfile.ZipFile('windows.zip', 'r') as zip_ref:
                zip_ref.extractall(OpenSimAD_DIR)
            os.remove('windows.zip')
        cmd1 = 'cmake "' + pathBuildExpressionGraphOS + '"  -A x64 -DTARGET_NAME:STRING="' + filename + '" -DSDK_DIR:PATH="' + SDK_DIR + '" -DCPP_DIR:PATH="' + CPP_DIR + '"'
        cmd2 = "cmake --build . --config RelWithDebInfo"
        
    elif os_system == 'Linux':
        pathBuildExpressionGraphOS = os.path.join(pathBuildExpressionGraph, 'linux')
        OpenSimADOS_DIR = os.path.join(OpenSimAD_DIR, 'linux')
        # Download libraries if not existing locally.
        if not os.path.exists(os.path.join(OpenSimAD_DIR, 'linux', 'lib')):
            url = 'https://sourceforge.net/projects/opensimad/files/linux.tar.gz'
            zipfilename = 'linux.tar.gz'                
            download_file(url, zipfilename)
            cmd_tar = 'tar -xf linux.tar.gz -C "{}"'.format(OpenSimAD_DIR)
            os.system(cmd_tar)
            os.remove('linux.tar.gz')
        cmd1 = 'cmake "' + pathBuildExpressionGraphOS + '" -DTARGET_NAME:STRING="' + filename + '" -DSDK_DIR:PATH="' + OpenSimADOS_DIR + '" -DCPP_DIR:PATH="' + CPP_DIR + '"'
        cmd2 = "make"
        BIN_DIR = pathBuild
        
    elif os_system == 'Darwin':
        pathBuildExpressionGraphOS = os.path.join(pathBuildExpressionGraph, 'macOS')
        OpenSimADOS_DIR = os.path.join(OpenSimAD_DIR, 'macOS')
        # Download libraries if not existing locally.
        if not os.path.exists(os.path.join(OpenSimAD_DIR, 'macOS', 'lib')):
            url = 'https://sourceforge.net/projects/opensimad/files/macOS.tgz'
            zipfilename = 'macOS.tgz'                
            download_file(url, zipfilename)
            cmd_tar = 'tar -xf macOS.tgz -C "{}"'.format(OpenSimAD_DIR)
            os.system(cmd_tar)
            os.remove('macOS.tgz')
        cmd1 = 'cmake "' + pathBuildExpressionGraphOS + '" -DTARGET_NAME:STRING="' + filename + '" -DSDK_DIR:PATH="' + OpenSimADOS_DIR + '" -DCPP_DIR:PATH="' + CPP_DIR + '"'
        cmd2 = "make"
        BIN_DIR = pathBuild
    
    os.chdir(pathBuild)    
    os.system(cmd1)    
    os.system(cmd2)
    
    if os_system == 'Windows':
        os.chdir(BIN_DIR)
        path_EXE = os.path.join(pathBuild, 'RelWithDebInfo', filename + '.exe')
        cmd2w = '"{}"'.format(path_EXE)
        os.system(cmd2w)
    
    # %% Part 2: build external function (i.e., build .dll).
    fooName = "foo.py"
    pathBuildExternalFunction = os.path.join(pathMain, 'buildExternalFunction')
    path_external_filename_foo = os.path.join(BIN_DIR, fooName)
    path_external_functions_filename_build = os.path.join(pathMain, 'build-ExternalFunction' + filename)
    path_external_functions_filename_install = os.path.join(pathMain, 'install-ExternalFunction' + filename)
    os.makedirs(path_external_functions_filename_build, exist_ok=True) 
    os.makedirs(path_external_functions_filename_install, exist_ok=True)
    shutil.copy2(path_external_filename_foo, pathBuildExternalFunction)
    
    sys.path.append(pathBuildExternalFunction)
    os.chdir(pathBuildExternalFunction)
    
    generateF(nInputs)
    
    if os_system == 'Windows':
        cmd3 = 'cmake "' + pathBuildExternalFunction + '" -A x64 -DTARGET_NAME:STRING="' + filename + '" -DINSTALL_DIR:PATH="' + path_external_functions_filename_install + '"'
        cmd4 = "cmake --build . --config RelWithDebInfo --target install"
    elif os_system == 'Linux':
        cmd3 = 'cmake "' + pathBuildExternalFunction + '" -DTARGET_NAME:STRING="' + filename + '" -DINSTALL_DIR:PATH="' + path_external_functions_filename_install + '"'
        cmd4 = "make install"
    elif os_system == 'Darwin':
        cmd3 = 'cmake "' + pathBuildExternalFunction + '" -DTARGET_NAME:STRING="' + filename + '" -DINSTALL_DIR:PATH="' + path_external_functions_filename_install + '"'
        cmd4 = "make install"
    
    os.chdir(path_external_functions_filename_build)
    os.system(cmd3)
    os.system(cmd4)    
    os.chdir(pathMain)
    
    if os_system == 'Windows':
        shutil.copy2(os.path.join(path_external_functions_filename_install, 'bin', filename + '.dll'), CPP_DIR)
    elif os_system == 'Linux':
        shutil.copy2(os.path.join(path_external_functions_filename_install, 'lib', 'lib' + filename + '.so'), CPP_DIR)
        os.rename(os.path.join(CPP_DIR, 'lib' + filename + '.so'), os.path.join(CPP_DIR, filename + '.so'))
    elif os_system == 'Darwin':
        shutil.copy2(os.path.join(path_external_functions_filename_install, 'lib', 'lib' + filename + '.dylib'), CPP_DIR)
        os.rename(os.path.join(CPP_DIR, 'lib' + filename + '.dylib'), os.path.join(CPP_DIR, filename + '.dylib'))
    
    os.remove(os.path.join(pathBuildExternalFunction, "foo_jac.c"))
    os.remove(os.path.join(pathBuildExternalFunction, fooName))
    os.remove(path_external_filename_foo)
    shutil.rmtree(pathBuild)
    shutil.rmtree(path_external_functions_filename_install)
    shutil.rmtree(path_external_functions_filename_build)    

# %% From storage file to numpy array.
def storage2numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    return data

# %% From storage file to DataFrame.
def storage2df(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out

# %% From numpy array to storage file.
def numpy2storage(labels, data, storage_file):
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"

    f = open(storage_file, 'w')
    f.write('name %s\n' % storage_file)
    f.write('datacolumns %d\n' % data.shape[1])
    f.write('datarows %d\n' % data.shape[0])
    f.write('range %f %f\n' % (np.min(data[:, 0]), np.max(data[:, 0])))
    f.write('endheader \n')

    for i in range(len(labels)):
        f.write('%s\t' % labels[i])
    f.write('\n')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' % data[i, j])
        f.write('\n')

    f.close()
    
# %% Download file given url.
def download_file(url, file_name):
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
