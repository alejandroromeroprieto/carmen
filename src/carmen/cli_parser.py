"""
CLI entry point for the carbon cycle emulator.
"""

import argparse


def cli_parser():
    """
    CLI parser for the carbon cycle model emulator.

    Basic usage looks like this:

        PYTHONPATH=src python -m carmen.carmen UKESM1-0-LL

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

    --store_results      : Boolean flag that, if present, instucts the emulator to store
                           the results from the emulation.

    """
    # Supported models
    model_list = {
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
    }
    scenario_opts = [
        "abrupt-4xCO2",
        "1pctco2",
        "1pctco2-cdr",
        "ssp119",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp434",
        "ssp460",
        "ssp534-over",
        "ssp585",
        "hist-noLu",
        "flat10",
        "esm-ssp370",
        "esm-ssp119",
        "esm-ssp126",
        "esm-ssp245",
        "esm-ssp534-over",
        "esm-ssp585",
    ]
    parser = argparse.ArgumentParser()

    parser.add_argument("models", nargs="+")
    parser.add_argument("-s", "--scenario", choices=scenario_opts, default="ssp585")
    parser.add_argument("-p", "--scenario_pars", choices=scenario_opts, default=None)
    parser.add_argument("-r", "--realisation", default="default")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    parser.add_argument("--npp", action=argparse.BooleanOptionalAction)
    parser.add_argument("--store_results", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.scenario_pars is None:
        args.scenario_pars = "cross_experiment"

    for model in args.models:
        if model not in model_list["CMIP6_" + args.scenario_pars]:
            raise ValueError(
                f"Wrong model name '{model}' for "
                f"CMIP6 era and {args.scenario_pars} scenario. "
                f"Available models are: {model_list['CMIP6_' + args.scenario_pars]}"
            )

    return (
        args.models,
        args.scenario,
        args.scenario_pars,
        args.realisation,
        args.save,
        args.npp,
        args.store_results,
    )
