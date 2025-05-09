# Pescoid Simulation Basics
The one-dimensional active fluid dynamic simulation is run using a finite element differential equation solver, FENICS. We are using the legacy version 2019.1.0, not FENICSX, which has some slight differences. In the requirement file I have outlined what is necessary to run the program as well as some python packages that improve quality of life.

# Scipt Purposes
1. 'nondim_pesc_v14.py' is core of the simulation. It produced results for a given parameter set and then saves them into '_pycache_'. This is the script that we would like to have processed in parallel, where each core handles a subset of the finite element mesh.
2. 'state_diagram_v5.py' is meant to run the simulation over a range of parameter values and save those values into a cach. If we can parallel process such that we cover different parts of the parameter values range with different cores, that would be ideal.
3. 'pescoid_plotting_v7.py' is the plotting code for individual simulations
4. 'trial_sim.py' allows you to run an individual simulation