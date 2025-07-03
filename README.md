<div align="center">
  <h1>1-Dimensional active fluid model of tissue wetting and dewetting</h1>
</div>

<div align="center">
    <img src='docs/_static/pescoid_mesoderm_animation.gif'
    style="width:80%">
</div>
<div align="center">
    <img src='docs/_static/pescoid_1d_profiles_animation.gif'
    style="width:80%">
</div>
<br>

A computational model simulating tissue morphogenesis as an active fluid system in one dimension. This project implements coupled partial differential equations (PDEs) that capture:

* Tissue growth and density dynamics
* Mesoderm differentiation with feedback mechanisms
* Mechanical force balance with active stresses

The simulation is implemented using FEniCS for finite element PDE solving. Both a semi-brute force grid-search and CMA-ES are used for parameter optimization to fit the model to reference data.

## Installation
This package utilizes [legacy FENICS](https://fenicsproject.org/download/archive/). To install an appropriate environment:
```sh
# Create environment with legacy FEniCS
conda create -n pescoid python=3.11 -y
conda install -n pescoid -c conda-forge fenics -y

# Install required packages
conda activate pescoid
pip install -r requirements.txt
```

This package is currently in active developlment. If you'd like to use it to test or run simulations, install it in editable mode via:
```sh
git clone https://github.com/oblanch/PescoidProject.git
cd pescoid-modelling
pip install -e .
```


## Runtime requirements
Simulation and optimization runs require different sets of parameters. An example for simulations is provided in [`configs/x0_simulation.yaml`](configs/x0_simulation.yaml) and an example for optimization is provided in [`configs/optimization_config.yaml`](configs/optimization_config.yaml).

A reference timeseries with data for tissue size and mesoderm fraction is used for the optimization as well as post-simulation comparison. We provide this data in [`data/reference_timeseries.npz`](data/reference_timeseries.npz). *The reference file contains smoothed and plateau-adjusted trajectories derived from the experimental measurements using the [`make_reference_timeseries`](pescoid_modelling/utils/helpers.py#L11) function. For raw experimental data, please see the [Zenodo repository](https://zenodo.org/record/YOUR_RECORD_ID).*

## Examples
### Run simulations:
```shell
# single simulation
pescoid simulate \
  --config path/to/your/simulation_config.yaml \
  --output_dir path/to/output
```

**Optional arguments:**

* `--generate_figures`: Plot simulation trajectory against experimental trajectory
* `--experimental_npz [PATH/TO/.NPZ]`: Specify reference timeseries for `--generate_figures`
* `--name [NAME]`: Choose a folder output name (defaults to config.yaml stem)

Users can override any of the parameters specified in their simulation YAML by supplying it as an argument. For example:
* `--activity 0.5 --r 5.0 --tau_m 10.0`

<br>

### Run parameter optimization:
```shell
# full optimization
pescoid optimize \
  --config path/to/your/optimization_config.yaml \
  --output_dir path/to/output \
  --experimental_npz path/to/reference_timeseries.npz
```

**Optional arguments:**
* `--optimization_target [TARGET]`: Choose between `["tissue", "mesoderm", "tissue_and_mesoderm"]`. Adjusts the fitness objective to optimize for individual trajectories or both. Defaults to `tissue_and_mesoderm`.

<br>

### Standalone visualization:
```shell
# figures only
pescoid plot \
  --simulation_npz path/to/simulation_results.npz \
  --experimental_npz path/to/reference_timeseries.npz \
  --output_dir path/to/place/figure
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
