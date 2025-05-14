"""
CLI entry point for the carbon cycle emulator.
"""

import argparse


def cli_parser():
    """
    CLI parser for the carbon cycle model emulator.

    Basic usage looks like this:

        python carbon_cycle.py UKESM1-0-LL

    Where the desired model (e.g. UKESM1-0-LL) to emulate is the only positional
    argument. That command defaults to running the SSP585 scenario from a set of
    parameters calibrated using all scenarios for that specific model.

    This base behaviour can be modified by the following optional arguments:

    --scenario, -s  :      Changes the data used to run the simulation to the desired
                           scenario. Notice that the data should come from the same model
                           we are emulating, so this option is limited to all the SSP
                           scenarios that are published for the given CMIP6 model.

    --scenario_pars, -p  : Changes the parameter set used to run the model. In particular,
                           it instructs the emulator to use the parameters derived by
                           calibrating a specific SSP scenario. Notice that this should
                           still be linked to the same model the emulator is trying to
                           emulate.

    --save,              : save some results from the emulation.

    --npp                : Boolean flag that, if present, instucts the emulator to use
                           NPP, rather than GPP and vegetation respiration separately.

    """
    # Supported models
    model_list = {
        # "CMIP6_1pctco2": [
        #     "ACCESS-ESM1-5",
        #     "BCC-CSM2-MR",
        #     "CanESM5",
        #     "CESM2",
        #     "CMCC-ESM2",
        #     "CNRM-ESM2-1",
        #     "IPSL-CM6A-LR",
        #     "MIROC-ES2L",
        #     "MPI-ESM1-2-LR",
        #     "MRI-ESM2-0",
        #     "NorESM2-LM",
        #     "UKESM1-0-LL",
        # ],
        # "CMIP6_1pctco2-cdr": [
        #     "ACCESS-ESM1-5",
        #     "CanESM5",
        #     "CESM2",
        #     "CNRM-ESM2-1",
        #     "MIROC-ES2L",
        #     "NorESM2-LM",
        #     "UKESM1-0-LL",
        # ],
        # "CMIP6_ssp126": ["UKESM1-0-LL"],
        # "CMIP6_ssp245": ["UKESM1-0-LL"],
        # "CMIP6_ssp434": ["UKESM1-0-LL"],
        # "CMIP6_ssp534-over": ["UKESM1-0-LL"],
        # "CMIP6_ssp585": [
        #     "UKESM1-0-LL",
        #     "IPSL-CM6A-LR",
        #     "MPI-ESM1-2-LR",
        #     "MIROC-ES2L",
        #     "MRI-ESM2-0",
        # ],
        # "CMIP6_historical": ["IPSL-CM6A-LR"],
        # These parameter values were trained using all avaiable SSPs for the model
        "CMIP6_cross_experiment": [
            "ACCESS-ESM1-5",
            "BCC-CSM2-MR",
            "CanESM5",
            "CESM2",
            "CMCC-ESM2",
            "CNRM-ESM2-1",
            "GFDL-ESM4",
            "IPSL-CM6A-LR",
            "MIROC-ES2L",
            "MPI-ESM1-2-LR",
            "MRI-ESM2-0",
            "NorESM2-LM",
            "UKESM1-0-LL",
        ],
        # "CMIP6_holistic": [
        #     "UKESM1-0-LL",
        #     "MPI-ESM1-2-LR",
        #     "IPSL-CM6A-LR",
        #     "MIROC-ES2L",
        #     "NorESM2-LM",
        #     "CanESM5",
        #     "MRI-ESM2-0",
        #     "CNRM-ESM2-1",
        #     "CESM2",
        #     "BCC-CSM2-MR",
        #     "ACCESS-ESM1-5",
        #     "CMCC-ESM2",
        #     "GFDL-ESM4",
        # ],
    }
    scenario_opts = [
        "abrupt-4xCO2",
        "1pctco2",
        # "1pctco2-cdr",
        "ssp119",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp434",
        "ssp460",
        "ssp534-over",
        "ssp585",
        # "manual-ssps",
        # "manual-1pctco2",
        # "historical",
        # "cross_experiment",
        # "holistic",
        "hist-noLu",
    ]
    parser = argparse.ArgumentParser()

    parser.add_argument("models", nargs="+")
    parser.add_argument("-s", "--scenario", choices=scenario_opts, default="ssp585")
    parser.add_argument("-p", "--scenario_pars", choices=scenario_opts, default=None)
    parser.add_argument("-r", "--realisation", default="default")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    parser.add_argument("--npp", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.scenario_pars is None:
        args.scenario_pars = "cross_experiment"

    for model in args.models:
        if model not in model_list["CMIP6_" + args.scenario_pars]:
            raise ValueError(
                f"Wrong model name '{model}' for "
                f"CMIP6 era and {args.scenario_pars} scenario. "
                f"Available models are: {model_list["CMIP6_" + args.scenario_pars]}"
            )
    # TODO: maybe have a table somewhere in git with all available models so I can just
    # TODO: link to it?

    return args.models, args.scenario, args.scenario_pars, args.realisation, args.save, args.npp
