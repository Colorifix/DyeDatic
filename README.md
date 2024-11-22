## DyeDactic: learning colours
--- -
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A colour prediction workflow for biosynthetically produced dyes and pigments.  
A code repository to reproduce the results published by [Karlov et al.]()

## Command line set up
- [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) is required to run the package I used version (1.8.3)
- Clone the repo `git clone git@github.com:Colorifix/dyedactic_public.git`
- Download a release of [XTB executable](https://xtb-docs.readthedocs.io/en/latest/setup.html) (6.6.1 was tested) and make sure executable is in your $PATH
- Make sure you are in the root directory `cd DyeDactic_public`
- Please run `poetry install` to install dependencies
- Then any script can be launched using `poetry run python /path/to/script.py`

## Description of package files
### Data and source files
- `src/` - directory contains all functions and classes from 3D structure generation to colour estimation. 
- `src/convert_spectrum_to_colour.py` - contains functions to convert absorption spectra to RGB colours together with a test run
- `src/optimize_xtb.py` - has wrapping functions to use XTB external program to optimise geometry, calculate energies, and HOMO-LUMO gaps  
- `src/orca_inputs.py` - a class for generating [ORCA](https://www.faccts.de/orca/) (5.0.3) input files to optimise geometry and do TD-DFT calculations 
- `src/tauromers.py` - tautomer enumeration functions which use euristics to prune tautomer generation tree and estimate energies using XTB energies
- `src/utils.py` - miscellaneous helper functions
- `data/` - contains the collected data set of natural colourants, calculated QM descriptors, TD-DFT results, and experimental absorption spectra 
- `data/pigments.csv` - a csv file containing the database of collected natural compounds with experimental data and references
- `data/pigment_pH_SI.csv` - contains experimental absorption spectra for 4 colourants explored in the paper (*emodin*, *quinalizarin*, *biliverdin*, and *orcein*) at different pH levels
- `inputs/` - directory for input files for ORCA calculations
- `mpnn_training/` - a specified folder for chemprop based neural network model to predict absorption lowest light absorption energies
- `mpnn_training/data/` - a directory for raw and clean training data
- `mpnn_training/data/20210205_all_expt_data_no_duplicates_solvent_calcs.csv` - a training set provided by [Greenman et al.](https://doi.org/10.1039/D1SC05677H)
- `mpnn_training/data/data_all.csv` - a csv file for MPNN training data (90% of natural and 90% of artificial colourants together after split), validation, and test data sets with transitions and solvent
- `mpnn_training/data/test_natural.csv` - a natural colourant test set (10% of the collected set) solely to estimate prediction error
- `mpnn_training/hyperopt` - parameters and NN weights
- `mpnn_training/chemprop_hyperopt.py` a script to run hyperparameter optimisation (training is done using GPU)
- `mpnn_training/predict.py` - a prediction script which runs with test_natural.csv by default
- `mpnn_training/prepare_dataset.ipynb` - prepare a train/test spilt for MPNN training and clean the initial data from outliers; to run the natural compound split TD-DFT calculations hav to be done

### Scripts for image generation and input preparation
- `biliverdin_colour_vs_pH.py` - a script for *biliverdin* halochromicity visualisation based pKa, transition energies and oscillator strengths 
- `emodin_colour_vs_pH.py` - a script for *emodin* halochromicity visualisation based pKa, transition energies and oscillator strengths 
- `quinalizarin_colour_vs_pH.py` - a script for *quinalizarin* halochromicity visualisation based pKa, transition energies and oscillator strengths 
- `orcein_colour_vs_pH.py` - a script for *orcein* halochromicity visualisation based pKa, transition energies and oscillator strengths 
- `generate_inputs.py` - a script to generate inputs files for ORCA and xyz coordinates
- `plot_experimental_spectra_SI.py` - takes experimental spectra in csv format, prints, and converts to corresponding colours
