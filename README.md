# OpenSimAD
Windows libraries for OpenSimAD - OpenSim with support for Algorithmic Differentiation.

## How to generate an external function for use with CasADi?
OpenSimAD is used to formulate trajectory optimization problems with OpenSim musculoskeletal models. To leverage the benefits of algorithmic differentiation, we use [CasADi external functions](https://web.casadi.org/docs/#casadi-s-external-function). In our case, the external functions typically take as inputs the multi-body model states (joint positions and speeds) and controls (join accelerations) and return the joint torques after solving inverse dynamics. The external functions can them be called when formulating trajectry optimization problems (e.g., https://github.com/antoinefalisse/3dpredictsim and https://github.com/antoinefalisse/predictsim_mtp).

Here we provide code and examples to generate external functions automatically given an OpenSim musculoskeletal model (.osim file).

### Install requirements
  - Third-party software:
    - CMake
    - Visual studio (test with Visual Studio 2017 Community)
    - Anaconda
  - conda environment:
    - Open Anaconda prompt
    - Create environment: `conda create -n opensimAD pip spyder python=3.8`
    - Activate environment: `activate opensimAD`
    - Navigate to the folder where you want to download the code: eg. `cd Documents`
    - Download code: `git clone https://github.com/antoinefalisse/opensimAD.git`
    - Navigate to the folder: `cd opensimAD`
    - Install required packages: `python -m pip install -r requirements.txt`
    - Install OpenSim by following the instructions [here](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Scripting+in+Python)
 


## Citation
Please cite this paper in your publications if OpenSimAD helps your research:
  - Falisse A, Serrancolí G, et al. (2019) Algorithmic differentiation improves the computational efficiency of OpenSim-based trajectory optimization of human movement. PLoS ONE 14(10): e0217730. https://doi.org/10.1371/journal.pone.0217730

Please cite this paper in your publications if you used OpenSimAD for simulations of human walking:
  - Falisse A, et al. (2019) Rapid predictive simulations with complex musculoskeletal models suggest that diverse healthy and pathological human gaits can emerge from similar control strategies. J. R. Soc. Interface.162019040220190402. http://doi.org/10.1098/rsif.2019.0402
