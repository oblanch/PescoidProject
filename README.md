# Pescoid fluid dynamics simulation
A computational model simulating tissue morphogenesis as an active fluid system in one dimension. This project implements coupled partial differential equations (PDEs) that capture:

* Tissue growth and density dynamics
* Mesoderm differentiation with feedback mechanisms
* Mechanical force balance with active stresses

The simulation is implemented using FEniCS for robust finite element PDE solving and CMA-ES for parameter optimization to fit the model to experimental data.

## Installation
This package utilizes legacy FENICS. To install an appropriate environment:
```sh
# Create environment with legacy FEniCS
conda create -n pescoid python=3.11 -y
conda install -n pescoid -c conda-forge fenics -y

# Install required packages
conda activate pescoid
pip install -r requirements.txt
```

This package is currently in active developlment. If you'd like to use it to test/run simulations, install it in editable mode via:
```sh
git clone https://github.com/oblanch/PescoidProject.git
cd pescoid-modelling
pip install -e .
```

## Runtime requirements
Each simulation depends on a set of parameters specified in `optimization_config.yaml`. Users can specify parameters for individual simulation runs, or upper and lower bounds for optimization. Simulation results get saved according to the yaml prefix, so specifying different configs allows an efficient and reproducible method for repeat runs.

## Examples
Run simulations with:
```sh
# single simulation
pescoid simulate \
  --config path/to/your/optimization_config.yaml \
  --output_dir sim_out/
```

Run CMA-ES optimization with:
```sh
# full optimization
pescoid optimize \
  --config path/to/your/optimization_config.yaml \
  --output_dir opt_out/ \
  --experimental_npz path/to/experimental_timeseries.npz
```

## Citation
```bibtex
@article{Kadiyala/Yang/2025,
  author       = {Kadiyala, U.*, Blanchard, O.*, Marschlich, N.*, et al.},
  title        = {Active wetting and dewetting dynamics of zebrafish embryonic explants},
  year         = {2025},
  doi          = {Coming soon},
  url          = {Coming soon},
  note         = {Preprint}
}
