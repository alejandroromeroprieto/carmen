NOTE: This is just an archive for the model at a given point in time (13/08/2025), when the semi-definitive structure of the model was decided but many of old calibrations and functions were still present. After this point there was a considerable effort to clean up the code. We are only keeping this as an archive for the short/medium term in case we need some of the information that was lost with the cleanup, but this branch will probably be deleted at some point when it is decided the archive is not needed anymore.
# carbon-cycle-model

TODO: write README

Run: PYTHONPATH=src python -m carbon_cycle_model.carbon_cycle_model UKESM1-0-LL --npp

Calibrate: PYTHONPATH=src python -m carbon_cycle_model.calibration.calibrate_model

See example_simple_usage.py