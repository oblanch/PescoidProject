from itertools import product
import numpy as np
import pandas as pd
from tqdm import tqdm
from nondim_pesc_v10_active_stress import run_non_dim_sim
import os


F_values = np.linspace(3e-1,3e-1, 1).round(1)
gamma_values = np.linspace(1e-2,1e-2,1)
T_m_values = np.linspace(6.0, 6.0, 1).round(1)
A_values = np.linspace(1.0, 1.5, 21 ).round(3)
delta_values = np.linspace(1e-3, 1e-3, 1).round(3)
length_scale_values = np.linspace(1.0, 1.0, 1).round(1)
r_values = np.linspace(1.0, 1.0, 1).round(1)
Beta_values = np.linspace(0.0,0.5,21).round(3)
Sigma_c_values = np.linspace(0.0,0.0,1).round(1)
delta_t_values = np.linspace(0.1,0.1,1)

print(f'dt: {delta_t_values}\nf: {F_values}\ngamma: {gamma_values}\nT_m: {T_m_values}\nA: {A_values}\ndelta: {delta_values}\nr: {r_values}\nbeta: {Beta_values}\nsigma_c: {Sigma_c_values}')

if input("are you sure? (y/n)") != "y":
    exit()


def find_transition(time_data, boundary_positions):
    max_position = np.max(boundary_positions)
    max_index = np.argmax(boundary_positions)
    max_time = time_data[max_index]
    return max_time, max_position

def find_relative_max_m(time_data, max_time, meso_frac_data):
    max_m_position = np.max(meso_frac_data)
    max_m_index = np.argmax(meso_frac_data)
    max_m_time = time_data[max_m_index]
    if np.max(meso_frac_data) <= 0:
        relative_max_m_time = np.nan
    else:
        relative_max_m_time = max_m_time - max_time
    return relative_max_m_time, max_m_time, max_m_position

def is_valid_simulation(aborted):
    if aborted == True:
        return False
    if aborted == False:
        return True
def determine_state(transition_time, transition_position, final_position, initial_position, time_bound):
    if np.isnan(transition_time) or np.isnan(transition_position):
        return 3
    elif transition_time >= time_bound or transition_position <= 1.1 * final_position:
        return 0
    elif transition_time < 60 or transition_position <= 1.1 * initial_position:
        return 2
    else:
        return 1

def generate_key(F, gamma, T_m, A, delta, length_scale, r, beta, sigma_c, dt):
    return f"F_{F}_gamma_{gamma}_Tm_{T_m}_A_{A}_delta_{delta}_ls_{length_scale}_r_{r}_beta_{beta}_sigma_c_{sigma_c}_delta_t_{dt}"

# Load cached results if they exist
cached_results = {}
cache_file = 'phase_diagram_results.csv'

if os.path.exists(cache_file):
    previous_results_df = pd.read_csv(cache_file)
    print("CSV Columns:", previous_results_df.columns)  # Debug: Check column names
    for _, row in previous_results_df.iterrows():
        # Ensure all parameters are passed, including length_scale if necessary
        key = generate_key(
            row.get('F'), row.get('gamma'), row.get('T_m'), row.get('A'), row.get('delta'),
            1.0,  # Assuming length_scale is 1.0, as per your ranges
            row.get('r'), row.get('beta'), row.get('sigma_c'), row.get('dt')
        )
        cached_results[key] = row.to_dict()

results_list = []

for F, gamma, T_m, A, delta, length_scale, r, beta, sigma_c, dt in tqdm(
    product(F_values, gamma_values, T_m_values, A_values, delta_values, length_scale_values, r_values, 
            Beta_values, Sigma_c_values, delta_t_values),
    total=len(F_values) * len(gamma_values) * len(T_m_values) * len(A_values) * len(delta_values) * 
          len(length_scale_values) * len(r_values) * len(Beta_values) * len(Sigma_c_values) * len(delta_t_values)
):

    key = generate_key(F, gamma, T_m, A, delta, length_scale, r, beta, sigma_c, dt)

    if key in cached_results:
        results_list.append(cached_results[key])
        continue

    # Run simulation
    boundary_positions, boundary_time_data, density_data, mesoderm_data, meso_frac_data, x_coords, aborted, dt = run_non_dim_sim(
        length_scale=length_scale, Delta=delta, Flow=F, Gamma=gamma, tau_m=T_m, Activity=A, R=r, Beta=beta, Sigma_c=sigma_c, delta_t = dt)

    final_position = boundary_positions[-1] if len(boundary_positions) > 0 else np.nan
    initial_position = boundary_positions[0]
    time_bound = boundary_time_data[-1] if len(boundary_time_data) > 0 else np.nan

    valid = is_valid_simulation(aborted)
    transition_time, transition_position = (np.nan, np.nan)
    relative_max_m_time = (np.nan)
    if valid:
        transition_time, transition_position = find_transition(boundary_time_data, boundary_positions)
        relative_max_m_time = find_relative_max_m(boundary_time_data, transition_time, meso_frac_data)[0]

    state = determine_state(transition_time, transition_position, final_position, initial_position, time_bound)

    result = {
        'dt': dt,
        'F': F,
        'gamma': gamma,
        'T_m': T_m,
        'A': A,
        'delta': delta,
        'r': r,
        'beta': beta,
        'sigma_c': sigma_c,
        'transition_position': transition_position,
        'transition_time': transition_time,
        'mesoderm_lag': relative_max_m_time,
        'state': state,
        'valid': valid
    }

    results_list.append(result)
    cached_results[key] = result

# Create DataFrame from the list of results
results_df = pd.DataFrame(results_list)

# Save results to CSV
results_df.to_csv(cache_file, index=False)
print(results_df)