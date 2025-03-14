"""
Example script on how to use the carbon cycle emulator.

It does the following:
- Initialise the emulator from one of the saved configurations.
- Load data from the ESM corresponding to the loaded configuration.
- Run the emulator via its single step function, as long as the ESM data extends.
- Interpolate results back to the yearly resolution of the ESM data.
- Create some diagnostic plots about the results.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

from carbon_cycle_model.carbon_cycle_model import CarbonCycle
from carbon_cycle_model.utils import load_esm_data
from carbon_cycle_model.constants import SCEN_DIR

import pandas as pd
from collections import defaultdict

class TimeSeriesStore:
    def __init__(self):
        # Nested dictionary: {esm_model -> {scenario_run -> {variable -> DataFrame}}}
        self.data = defaultdict(lambda: defaultdict(dict))

    def add_timeseries(self, esm_model, scenario, variable, timeseries):
        """
        Add a time series to the store.

        Parameters:
        - esm_model (str): Name of the ESM model.
        - scenario (str): Name of the scenario run.
        - variable (str): The variable name.
        - timeseries (pd.DataFrame or pd.Series): The time series data.
        """
        if isinstance(timeseries, pd.Series):
            timeseries = timeseries.to_frame(name=variable)  # Convert Series to DataFrame
            
        self.data[esm_model][scenario][variable] = timeseries

    def get_timeseries(self, esm_model=None, scenario=None, variable=None):
        """
        Retrieve time series data based on filtering criteria.

        - If all three parameters are provided, returns a single DataFrame.
        - If one or more parameters are omitted, returns a dictionary.

        Parameters:
        - esm_model (str, optional): Filter by ESM model.
        - scenario (str, optional): Filter by scenario.
        - variable (str, optional): Filter by variable.

        Returns:
        - Pandas DataFrame if all three filters are provided.
        - Dictionary otherwise.
        """
        if esm_model and scenario and variable:
            return self.data.get(esm_model, {}).get(scenario, {}).get(variable, None)
        elif esm_model and scenario:
            return self.data.get(esm_model, {}).get(scenario, {})
        elif esm_model:
            return self.data.get(esm_model, {})
        else:
            return self.data

    def plot_timeseries(self, esm_model=None, scenario=None, variable=None, stop_plot=False):
        """
        Plots all time series matching the given filters.

        - If a single variable is selected, multiple time series are plotted on one graph.
        - If a single ESM model and scenario are selected, multiple variables are plotted.

        """

        timeseries_dict = self.get_timeseries(esm_model, scenario, variable)

        if isinstance(timeseries_dict, pd.DataFrame):  # Single time series case
            timeseries_dict.plot(title=f"{esm_model} - {scenario} - {variable}")
            plt.xlabel("Time")
            plt.ylabel(variable)
            plt.legend()
            if not stop_plot:
                plt.show()
        elif isinstance(timeseries_dict, dict):
            # plt.figure(figsize=(10, 6))
            if not scenario:
                for scenario_label, sce_dict in timeseries_dict.items():
                    for key, df in sce_dict.items():
                        if isinstance(df, pd.DataFrame):
                            for col in df.columns:
                                if variable and variable == col:
                                    if col == "cveg" or col == "csoil":
                                        plt.plot(df.index, df[col], linestyle="-", linewidth="3")
                                    elif col == "catm":
                                        plt.plot(df.index, df[col], linestyle="-", linewidth="3")
                                    elif col == "fcsa" or col =="oflux" or col =="fcva" or col == "carbon_increase":
                                        pass
                                    else:
                                        plt.plot(df.index, df[col], linewidth="3")
                                elif variable:
                                    pass
                                else:
                                    if col == "cveg" or col == "csoil":
                                        plt.plot(df.index, df[col], linestyle="--", color="black", linewidth="3")
                                    elif col == "catm":
                                        plt.plot(df.index, df[col], linestyle="-.",  color="black", linewidth="3")
                                    elif col == "fcsa" or col =="oflux" or col =="fcva" or col == "carbon_increase":
                                        pass
                                    else:
                                        plt.plot(df.index, df[col], linewidth="3",  color="black")
                        elif isinstance(df, pd.Series):
                            plt.plot(df.index, df, label=key)
            else:
                if not esm_model:
                    for model_label, scenarios  in timeseries_dict.items():
                        for scenario_label, timeseries_dict in scenarios.items():
                            if scenario == scenario_label:
                                line_style = category_styles[scenario]
                                for key, df in timeseries_dict.items():
                                    if isinstance(df, pd.DataFrame):
                                        for col in df.columns:
                                            if variable and variable == col:
                                                if col == "cveg" or col == "csoil" or col=="carbon_increase":
                                                    plt.plot(df.index, df[col], linestyle=line_style, label=model_label, linewidth="3", color=category_colors[model_label])
                                                elif col == "catm":
                                                    plt.plot(df.index, df[col], linestyle=line_style, label=model_label, linewidth="3", color=category_colors[model_label])
                                                else:
                                                    plt.plot(df.index, df[col], linestyle=line_style, label=model_label, linewidth="3", color=category_colors[model_label])
                                            elif variable:
                                                pass
                                            else:
                                                if col == "cveg" or col == "csoil" or col=="carbon_increase":
                                                    plt.plot(df.index, df[col], linestyle="--", linewidth="3")
                                                elif col == "catm":
                                                    plt.plot(df.index, df[col], linestyle="-.", linewidth="3")
                                                else:
                                                    plt.plot(df.index, df[col], linewidth="3")
                                    elif isinstance(df, pd.Series):
                                        plt.plot(df.index, df, label=key)                    
                else:
                    for key, df in timeseries_dict.items():
                        if isinstance(df, pd.DataFrame):
                            for col in df.columns:
                                if col == "cveg" or col == "csoil":
                                    plt.plot(df.index, df[col], label=f"{col}", linestyle="-", linewidth="3")
                                elif col == "catm":
                                    plt.plot(df.index, df[col], label=f"{col}", linestyle="-", linewidth="3")
                                elif col == "fcsa" or col =="oflux" or col =="fcva" or col == "carbon_increase":
                                        pass
                                else:
                                    plt.plot(df.index, df[col], label=f"{col}", linewidth="3")
                        elif isinstance(df, pd.Series):
                            plt.plot(df.index, df, label=key, linewidth="3")
            # plt.title(f"Timeseries Plot ({esm_model or 'All Models'} - {scenario or 'All Scenarios'} - {variable or 'All Variables'})", fontsize=30)
            # plt.title(f"Timeseries Plot ({esm_model or 'All Models'} - SSP534-over & SSP585 - {variable or 'All Variables'})", fontsize=30)
            plt.title(f"Timeseries Plot ({esm_model or 'All Models'} - {scenario or 'All Scenarios'})", fontsize=30)
            plt.xlabel("Time", fontsize=20)
            plt.ylabel("Value", fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            # plt.legent(fontsize=10)
            plt.legend(loc="lower left", mode="expand", ncol=6, fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.5)

            if not stop_plot:
                plt.show()

MODEL_NAME_LIST = {
                #    "ACCESS-ESM1-5": ["ssp126", "ssp245", "ssp370", "ssp534-over", "ssp585"],
                #    "BCC-CSM2-MR": ["ssp126", "ssp245", "ssp370", "ssp585"],
                #    "CanESM5": ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp534-over", "ssp585"],
                #    "CESM2": ["ssp126", "ssp245", "ssp370", "ssp585"],
                #    "CMCC-ESM2": ["ssp126", "ssp245", "ssp370", "ssp534-over", "ssp585"],
                #    "CNRM-ESM2-1": ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp534-over", "ssp585"],
                #    "GFDL-ESM4": ["ssp126", "ssp370"],
                #    "IPSL-CM6A-LR": ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp534-over", "ssp585"],
                #    "MIROC-ES2L": ["ssp119", "ssp126", "ssp245", "ssp370", "ssp534-over", "ssp585"],
                #    "MPI-ESM1-2-LR": ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"],
                #    "MRI-ESM2-0": ["ssp585"],
                #    "NorESM2-LM": ["ssp126", "ssp245", "ssp370", "ssp534-over", "ssp585"],
                   "UKESM1-0-LL": ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp534-over", "ssp585"]
                   }

categories = list(MODEL_NAME_LIST.keys())
colors = cm.get_cmap("tab20")(np.linspace(0, 1, len(categories)))
category_colors = dict(zip(categories, colors))

scenarios_list = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp470", "ssp534-over", "ssp585"]
# line_style_list = [".-", "4_2:2p", ".", "", ":", "-.", "--", "-"]
line_style_list = ["--", "-", "-", "-", "-", "-", "--", "-"]
category_styles = dict(zip(scenarios_list, line_style_list))

DT_MODEL = 0.05
DT_MODEL_OCEAN = 0.05

NUM_STEPS = int(1 / DT_MODEL)

if 1 % DT_MODEL != 0:
    print(
        "WARNING: model timestep is not an exact divisor of 1 year"
        " Bad things may happen due to timestep mismatch"
    )

store = TimeSeriesStore()

NPP_FLAG = True

for model, scenarios in MODEL_NAME_LIST.items():
    for scenario in scenarios:
        cc_emulator = CarbonCycle(
            {"model": model, "scenario": scenario},
            DT_MODEL,
            DT_MODEL_OCEAN,
            npp_flag=NPP_FLAG,
            **{},
        )

        # Load diagnosed ESM data from the data dir
        if model == "CNRM-ESM2-1" or model == "IPSL-CM6A-LR":
            scen_to_use = SCEN_DIR + "/detrended_wrt_decade"
        else:
            scen_to_use = SCEN_DIR

        data_file = Path(__file__).parent / scen_to_use / f"sce_{model}_{scenario}.txt"
        print("\nLoading ESM data from: ", data_file)

        esm_data = load_esm_data(
            data_file,
            recalc_emis=True,
            ninit=20,  # Because it's an SSP scenario
            smoothing_pars={"type": "savgol", "pars": [21, 3]},
        )

        cc_emulator.run_full_simulation()

        cc_emulator.interpolate_results(esm_data.time)

        # Find differences in timeseries
        # land box
        store.add_timeseries(model, scenario, "cveg", pd.Series(data=(cc_emulator.land.cveg - esm_data.cveg)*100/esm_data.cveg, index=esm_data.time, name="cveg"))
        store.add_timeseries(model, scenario, "csoil", pd.Series(data=(cc_emulator.land.csoil - esm_data.csoil)*100/esm_data.csoil, index=esm_data.time, name="csoil"))
        store.add_timeseries(model, scenario, "npp", pd.Series(data=(cc_emulator.land.npp - esm_data.npp)*100/esm_data.npp, index=esm_data.time, name="npp"))
        store.add_timeseries(model, scenario, "lit", pd.Series(data=(cc_emulator.land.lit - esm_data.lit)*100/esm_data.lit, index=esm_data.time, name="lit"))
        store.add_timeseries(model, scenario, "sres", pd.Series(data=(cc_emulator.land.sres - esm_data.rh)*100/esm_data.rh, index=esm_data.time, name="sres"))
        if esm_data.fcvegout is not None:
            store.add_timeseries(model, scenario, "fcva", pd.Series(data=(cc_emulator.land.fcva - esm_data.fcvegout)*100/esm_data.fcvegout, index=esm_data.time, name="fcva"))
        
        if esm_data.fcsoilout is not None:
            store.add_timeseries(model, scenario, "fcsa", pd.Series(data=(cc_emulator.land.fcsa - esm_data.fcsoilout)*100/esm_data.fcsoilout, index=esm_data.time, name="fcsa"))
        
        if esm_data.fcvegoutcsoilin is not None:
            store.add_timeseries(model, scenario, "fcvs", pd.Series(data=(cc_emulator.land.fcvs - esm_data.fcvegoutcsoilin)*100/esm_data.fcvegoutcsoilin, index=esm_data.time, name="fcvs"))

        if not NPP_FLAG:
            store.add_timeseries(model, scenario, "gpp", pd.Series(data=(cc_emulator.land.gpp - esm_data.gpp)*100/esm_data.gpp, index=esm_data.time, name="gpp"))
            store.add_timeseries(model, scenario, "vres", pd.Series(data=(cc_emulator.land.vres - esm_data.ra)*100/esm_data.ra, index=esm_data.time, name="vres"))

        # Ocean box
        store.add_timeseries(model, scenario, "carbon_increase", pd.Series(data=(cc_emulator.ocean.carbon_increase - np.cumsum(esm_data.oflux)), index=esm_data.time, name="carbon_increase"))
        store.add_timeseries(model, scenario, "oflux", pd.Series(data=(cc_emulator.ocean.oflux - esm_data.oflux)*100/esm_data.oflux, index=esm_data.time, name="oflux"))

        # General carbon cycle box
        store.add_timeseries(model, scenario, "catm", pd.Series(data=(cc_emulator.catm - esm_data.catm)*100/esm_data.catm, index=esm_data.time, name="catm"))


# # Only one variable for all scenarios in one model
# store.plot_timeseries(esm_model="UKESM1-0-LL", variable="npp")

# # All variables for one model and one scenario
store.plot_timeseries(esm_model="UKESM1-0-LL")

# # Only one variable for one scenario and all models
# store.plot_timeseries(scenario="ssp585", variable="catm")

# # All models - ssp585 - catm/cveg/csoil
# # ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp534-over", "ssp585"]
# variable_of_interest = "cveg"
# store.plot_timeseries(scenario="119", variable=variable_of_interest, stop_plot=True)
# store.plot_timeseries(scenario="ssp126", variable=variable_of_interest, stop_plot=True)
# store.plot_timeseries(scenario="ssp245", variable=variable_of_interest, stop_plot=True)
# store.plot_timeseries(scenario="ssp370", variable=variable_of_interest, stop_plot=True)
# store.plot_timeseries(scenario="ssp434", variable=variable_of_interest, stop_plot=True)
# store.plot_timeseries(scenario="ssp585", variable=variable_of_interest, stop_plot=True)
# # All models - ssp534-over - catm/cveg/csoil
# store.plot_timeseries(scenario="ssp534-over", variable=variable_of_interest)

# Do them!

