# OpenSimAD
Libraries for OpenSimAD - OpenSim with support for Algorithmic Differentiation.

## How to generate an external function for use with CasADi?
OpenSimAD is used to formulate trajectory optimization problems with OpenSim musculoskeletal models. To leverage the benefits of algorithmic differentiation, we use [CasADi external functions](https://web.casadi.org/docs/#casadi-s-external-function). In our case, the external functions typically take as inputs the multi-body model states (joint positions and speeds) and controls (joint accelerations) and return the joint torques after solving inverse dynamics. The external functions can then be called when formulating trajectory optimization problems (e.g., https://github.com/antoinefalisse/3dpredictsim and https://github.com/antoinefalisse/predictsim_mtp).

Here we provide code and examples to generate external functions automatically given an OpenSim musculoskeletal model (.osim file). Visit https://github.com/antoinefalisse/predsim_tutorial for a tutorial about how to use these external functions when formulating and solving trajectory optimization problems.

### Install requirements
1. Third-party packages
	- **Windows only**: Install [Visual Studio](https://visualstudio.microsoft.com/downloads/)
		- The Community variant is sufficient and is free for everyone.
		- During the installation, select the *workload Desktop Development with C++*.
		- The code was tested with the 2017, 2019, and 2022 Community editions.
	- **Linux only**: Install OpenBLAS libraries
		- `sudo apt-get install libopenblas-base`
2. Conda environment
	- Install [Anaconda](https://www.anaconda.com/)
	- Open Anaconda prompt
	- Create environment (python 3.9 recommended): `conda create -n opensim-ad python=3.9`
	- Activate environment: `conda activate opensim-ad`
	- Install OpenSim: `conda install -c opensim-org opensim=4.4=py39np120`
		- Test that OpenSim was successfully installed:
			- Start python: `python`
			- Import OpenSim: `import opensim`
				- If you don't get any error message at this point, you should be good to go.
			- You can also double check which version you installed : `opensim.GetVersion()`
			- Exit python: `quit()`
		- Visit this [webpage](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Conda+Package) for more details about the OpenSim conda package.
	- (Optional): Install an IDE such as Spyder: `conda install spyder`
	- Clone the repository to your machine: 
		- Navigate to the directory where you want to download the code: eg. `cd Documents`. Make sure there are no spaces in this path.
		- Clone the repository: `git clone https://github.com/antoinefalisse/opensimAD.git`
		- Navigate to the directory: `cd opensimAD`
	- Install required packages: `python -m pip install -r requirements.txt`

### Example
  - run `main.py`
      - You should get as output a few files in the example folder. Among them: `F.cpp`, `F.dll`, and `F_map.npy`. The .cpp file contains the source code of the external function, the .dll file is the [dynamically linked library](https://web.casadi.org/docs/#casadi-s-external-function) that can be called when formulating your trajectory optimization problem, the .npy file is a dictionnary that describes the outputs of the external function (names and indices).
      - More details in the comments of `main.py` about what inputs are necessary and optional.

### Limitations
  - Not all OpenSim models are supported:
    - Your model **should not have locked joints**. Please replace them with weld joints (locked joints would technically require having kinematic constraints, which is possible but makes the problem more complicated).
    - **Constraints will be ignored** (eg, coupling constraints).
    - **SimmSplines are not supported**, as their implementation in OpenSim is not really compatible with algorithmic differentiation. See how we replaced the splines of the [LaiArnold_modifed model](https://github.com/antoinefalisse/opensimAD/blob/main/examples/LaiArnold_modified.osim#L3564) with polynomials.
  - OpenSimAD does not support all features of OpenSim. **Make sure you verify what you are doing**. We have only used OpenSimAD for specific applications.

## Tutorial
  - You can find [here a tutorial](https://github.com/antoinefalisse/predsim_tutorial) describing how to generate a predictive simulation of walking. The tutorial describes all the steps required, including the use of OpenSimAD to general external functions for use when formulating the trajectory optimization problem underlying the predictive simulation. 

## Citation
Please cite this paper in your publications if OpenSimAD helps your research:
  - Falisse A, Serrancolí G, et al. (2019) Algorithmic differentiation improves the computational efficiency of OpenSim-based trajectory optimization of human movement. PLoS ONE 14(10): e0217730. https://doi.org/10.1371/journal.pone.0217730

Please cite this paper in your publications if you used OpenSimAD for simulations of human walking:
  - Falisse A, et al. (2019) Rapid predictive simulations with complex musculoskeletal models suggest that diverse healthy and pathological human gaits can emerge from similar control strategies. J. R. Soc. Interface.162019040220190402. http://doi.org/10.1098/rsif.2019.0402

## Source code
The OpenSimAD libraries were compiled from [here](https://github.com/antoinefalisse/opensim-core/tree/AD-recorder-work-py-install).
