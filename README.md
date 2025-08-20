# CARMEN

  

This is a python implementation of the Carbon Atmosphere Regulation Model with Ecosystem Nodes. Currently, 13 different calibrations are available to run the model, each emulating the carbon cycle dynamics of a specific ESM: ACCESS-ESM1-5, BCC-SCM2-MR, CanESM5, CESM2, CMCC-ESM2, CMCC-ESM2, CNRM-ESM2-1, GFDL-ESM4, IPSL-CM6A-LR, MIROC-ES2L, MPI-ESM1-2-LR, MRI-ESM2-0, NorESM2-LM, UKESM1-0-LL.

  
  

# Running

  

Once inside an environemnt with the required python libraries (see section below), there are two options to run CARMEN:

  

- Through the CLI tool: CARMEN implements a CLI tool to run the model interactively. From the base folder simply run: `PYTHONPATH=src python -m carbon_cycle_model.carbon_cycle_model <model> --scenario <scenario> --npp`, where `<model>` and `<scenario>` are the ESM and scenario you would like to emulate. For instance, `PYTHONPATH=src python -m carbon_cycle_model.carbon_cycle_model UKESM1-0-LL --scenario ssp434 --npp` will run an emulation of the carbon cycle from UKESM1-0-LL for the ssp434 scenario, and plot the results comparing CARMEN's emulation and the underlying ESM data.
- Programmatically: CARMEN has been designed to be easy to couple to other models. The only two external quantities it requires are annual global air temperature and annual global ocean temperature. The `src/carbon_cycle_model/example_simple_usage.py` script presents a minimalist example on how CARMEN can be used programmatically. 

# Calibration

To produce new parametrisations aiming to emulate different ESMs, the `src/carbon_cycle_model/calibration/calibrate_model.py` script should be used. This script will load ESM data from the `src/carbon_cycle_model/data/scenarios` and run a calibration routine for each carbon flux to minimise the difference between the ESM carbon fluxes and CARMEN's emulation. See the `src/carbon_cycle_model/data/scenarios` folder for the structure and naming scheme these ESM files should follow to be recognised by the calibration script. Once the ESM data has been provided, you can specify the ESM model name and scenarios to use in the calibration at the top of the `calibrate_model.py` script.
  

# Required environment
TODO: I have created some conda environment to run the model, but I need to wrtie down the requirements. 
  

# Repository structure

The main folders in the repository are:

- Model code: this lives under the `land_component` and `ocean_component` folders in `src/carbon_cycle_model/`. The `CarbonCycle` class providing CARMEN's interface and bringing the two components together lives in `src/carbon_cycle_model/carbon_cycle_model.py`.
- Calibration code: you can find all utils related to the calibration of CARMEN under `src/carbon_cycle_model/calibration`. 
- Example usage: you can find an example of how CARMEN can be imported and run programmatically in `src/carbon_cycle_model/example_simple_usage.py`
- Data: the `src/carbon_cycle_model/data` stores both annual ESM data used for CARMEN's calibration and the parameters resulting from that calibration.
- Plotting tools: the `src/carbon_cycle_model/plotting` folder stores some useful utils for plotting several metrics associated with CARMEN's emulation.
- Output: the `output` folder stores simulation data and plots from CARMEN's emulations, which are produced by various tools within the repository.
