# RSC-SIM Radio Astronomy Interference Analysis

This directory contains a series of python scripts and notebooks for validating observatory antenna beam models, modeling multiple satellite constellations with variable emissions described by probability distribution functions, and computing EPFD per ITU-R S.1586-1 for comparison to ITU thresholds for harmful interference described in ITU-R RA.769.2

<br />


## 📁 Directory Structure

```
educational_tutorials/
├── README.md                          # This file
├── 01_basic_telescope_model.py        # Basic Telescope Model Validation
|── shared/                            # Shared utilities
|   ├── __init__.py                    # Package initialization
|   ├── config.py                      # Configuration parameters
|   ├── instrument_setup.py            # Instrument configuration
|   ├── sky_models.py                  # Sky temperature models
|   └── plotting_utils.py              # Plotting functions
└──data/			       # Data required by functions in this 
    ├──table1_oxygen.csv
    ├──table2_water_vapor.csv
    
```
