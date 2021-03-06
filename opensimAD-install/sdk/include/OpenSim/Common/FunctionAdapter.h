#ifndef OPENSIM_FUNCTION_ADAPTER_H_
#define OPENSIM_FUNCTION_ADAPTER_H_
/* -------------------------------------------------------------------------- *
 *                        OpenSim:  FunctionAdapter.h                         *
 * -------------------------------------------------------------------------- *
 * The OpenSim API is a toolkit for musculoskeletal modeling and simulation.  *
 * See http://opensim.stanford.edu and the NOTICE file for more information.  *
 * OpenSim is developed at Stanford University and supported by the US        *
 * National Institutes of Health (U54 GM072970, R24 HD065690) and by DARPA    *
 * through the Warrior Web program.                                           *
 *                                                                            *
 * Copyright (c) 2005-2017 Stanford University and the Authors                *
 * Author(s): Peter Eastman                                                   *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may    *
 * not use this file except in compliance with the License. You may obtain a  *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.         *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 * -------------------------------------------------------------------------- */


// INCLUDES
#include "Function.h"


//=============================================================================
//=============================================================================
namespace OpenSim { 

// Excluding this from Doxygen until it has better documentation! -Sam Hamner
/// @cond
/**
 * This is a SimTK::Function that acts as a wrapper around an OpenMM::Function.
 *
 * @author Peter Eastman
 */
class OSIMCOMMON_API FunctionAdapter : public SimTK::Function
{
//=============================================================================
// DATA
//=============================================================================
protected:
    // REFERENCES
    /** The OpenSim::Function used to evaluate this function. */
    const OpenSim::Function& _function;

//=============================================================================
// METHODS
//=============================================================================
public:
    FunctionAdapter(const OpenSim::Function &aFunction);
	osim_double_adouble calcValue(const SimTK::Vector& x) const override;
	osim_double_adouble calcDerivative(const std::vector<int>& derivComponents, const SimTK::Vector& x) const;
	osim_double_adouble calcDerivative(const SimTK::Array_<int>& derivComponents, const SimTK::Vector& x) const override;
    int getArgumentSize() const override;
    int getMaxDerivativeOrder() const override;
private:
    FunctionAdapter();
    FunctionAdapter& operator=(FunctionAdapter& function);

//=============================================================================
};  // END class FunctionAdapter

/// @endcond

}; //namespace
//=============================================================================
//=============================================================================

#endif // OPENSIM_FUNCTION_ADAPTER_H_
