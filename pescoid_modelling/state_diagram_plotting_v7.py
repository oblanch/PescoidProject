from itertools import product
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap



# Define the custom colormap for states
cmap = ListedColormap(['yellow', 'blue', 'purple', 'red'])

# Load results from CSV
output_dir = 'simulation_results/state_diagrams/'
results_df = pd.read_csv('state_diagram_results.csv')
print(results_df)
def plot_state_diagram(results_df, focus_params, all_params, output_dir, focus_param_ranges=None, other_param_ranges=None):
    if len(focus_params) != 2:
        raise ValueError("Exactly two parameters must be selected to focus on.")

    # If no range is provided, use all values
    if focus_param_ranges:
        for param, param_range in focus_param_ranges.items():
            if param in focus_params:
                results_df = results_df[(results_df[param] >= param_range[0]) & (results_df[param] <= param_range[1])]

    if other_param_ranges:
        for param, param_range in other_param_ranges.items():
            if param in focus_params:
                results_df = results_df[(results_df[param] >= param_range[0]) & (results_df[param] <= param_range[1])]

    # Determine non-focus parameters
    non_focus_params = [p for p in all_params if p not in focus_params]

    # Group by combinations of non-focus parameters
    grouped = results_df.groupby(non_focus_params)

    for values, group in grouped:
        plt.figure(figsize=(8, 6))
        pivot_table = group.pivot_table(index=focus_params[0], columns=focus_params[1], values='state', aggfunc='mean')
        
        # Extract unique sorted values for ticks
        x_ticks = np.sort(group[focus_params[1]].unique())
        y_ticks = np.sort(group[focus_params[0]].unique())

        # Plot with custom colormap including four states
        plt.imshow(pivot_table, aspect='auto', cmap=cmap, vmin=0, vmax=3,
                   extent=[x_ticks.min(), x_ticks.max(), y_ticks.min(), y_ticks.max()], origin='lower')
        plt.colorbar(ticks=[0, 1, 2, 3], boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], label='State')

        plt.xticks(ticks=x_ticks, rotation=90)
        plt.yticks(ticks=y_ticks)

        plt.xlabel(focus_params[1].replace('_', ' ').title())
        plt.ylabel(focus_params[0].replace('_', ' ').title())
        title = f'State Diagram: {focus_params[0]} vs {focus_params[1]}'

        # Add plot title with unique non-focused parameter values
        extra_info = ', '.join([f"{param}={value}" for param, value in zip(non_focus_params, values)])
        plt.title(f"{title}\n({extra_info})")

        # Save plots named based on variable combinations
        filename = f'{output_dir}state_diagram_{focus_params[0]}_{focus_params[1]}_' + '_'.join([f"{param}_{value}" for param, value in zip(non_focus_params, values)]) + '.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
# Define all parameter names
parameter_names = ['F', 'gamma', 'T_m', 'A', 'delta', 'r', 'beta', 'sigma_c']

# Select focus parameters
focus_params = ['A', 'beta']

# Define range for the focus parameters, for example
focus_param_ranges = {
    'A': (1, 1.5),  # Example range for parameter A
    'beta': (0, 1)   # Example range for parameter r
}
other_param_ranges = {
}
# Generate plots with the specified range
plot_state_diagram(results_df, focus_params, parameter_names, output_dir, focus_param_ranges, other_param_ranges)