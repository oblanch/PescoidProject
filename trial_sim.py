import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from nondim_pesc_v14 import run_non_dim_sim
from pescoid_plotting_v5 import pescoid_plot


#results = run_non_dim_sim(length_scale = 1.0, Delta = 1e-3, Flow  = 3e-1, tau_m = 6e0, Gamma = 1e-2, Activity = 1.0e0, R = 1e0, Beta = 0.5, Sigma_c = 0.0, delta_t=0.1) #excellent experimental match
results = run_non_dim_sim(length_scale = 1.0, Delta = 1e-3, Flow  = 3e-1, tau_m = 6e0, Gamma = 1e-2, Activity = 1.15e0, R = 1e0, Beta = 0.45, Sigma_c = 0.0, delta_t = 0.1, feedback = 'strain rate')#excellent experimental match



pescoid_plot()
