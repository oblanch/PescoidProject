import os
from itertools import product, combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results from CSV
output_dir = 'simulation_results/transition_time_heatmaps/'
results_df = pd.read_csv('state_diagram_results.csv')
print(results_df)

def plot_transition_time_heatmap(results_df, focus_params, all_params, output_dir, result_type, focus_param_ranges=None, other_param_ranges=None):
    if len(focus_params) != 2:
        raise ValueError("Exactly two parameters must be selected to focus on.")

    # Filter based on parameter ranges, if provided
    if focus_param_ranges:
        for param, param_range in focus_param_ranges.items():
            if param in focus_params:
                results_df = results_df[(results_df[param] >= param_range[0]) & (results_df[param] <= param_range[1])]

    if other_param_ranges:
        for param, param_range in other_param_ranges.items():
            if param not in focus_params:
                results_df = results_df[(results_df[param] >= param_range[0]) & (results_df[param] <= param_range[1])]

    # Filter for state '1'
    state_1_data = results_df[results_df['state'] == 1]

    # Determine non-focus parameters
    non_focus_params = [p for p in all_params if p not in focus_params]

    # Group by combinations of non-focus parameters
    grouped = state_1_data.groupby(non_focus_params)

    for values, group in grouped:
        plt.figure(figsize=(8, 6))
        # Pivot table orders y-axis rows from top to bottom by default, flip it for increasing order
        pivot_table = group.pivot_table(index=focus_params[0], columns=focus_params[1], values=result_type, aggfunc='mean')

        # Sort the unique values in increasing order for ticks
        x_ticks = np.sort(pivot_table.columns.values)
        y_ticks = np.sort(pivot_table.index.values)
        
        label = ''
        if result_type == 'transition_time':
            label = 'Transition Time'

        if result_type == 'mesoderm_lag':
            label = 'Mesoderm Lag'
        if result_type == 'transition_position':
            label = 'Transition Position'

        # Plot heatmap with normalized values
        sns.heatmap(pivot_table, cmap='viridis', vmin=min(state_1_data[result_type]), vmax=max(state_1_data[result_type]), cbar_kws={'label': f'{label}'}, 
                    xticklabels=True, yticklabels=True)

        plt.gca().invert_yaxis()  # Ensure y-axis values increase from bottom to top

        plt.xticks(ticks=np.arange(len(x_ticks)), labels=x_ticks, rotation=90)
        plt.yticks(ticks=np.arange(len(y_ticks)), labels=y_ticks)

        plt.xlabel(focus_params[1].replace('_', ' ').title())
        plt.ylabel(focus_params[0].replace('_', ' ').title())
        
        
        title = f'{label}: {focus_params[0]} vs {focus_params[1]}'

        # Add plot title with unique non-focused parameter values
        extra_info = ', '.join([f"{param}={value}" for param, value in zip(non_focus_params, values)])
        plt.title(f"{title}\n({extra_info})")

        # Save plots named based on variable combinations
        filename = f'{output_dir}{result_type}_{focus_params[0]}_{focus_params[1]}_' + '_'.join([f"{param}_{value}" for param, value in zip(non_focus_params, values)]) + '.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

# Define all parameter names
parameter_names = ['F', 'gamma', 'T_m', 'A', 'delta', 'r', 'beta', 'sigma_c']

# Select focus parameters
focus_params = ['A', 'beta']

# Define range for the focus parameters, if necessary
focus_param_ranges = {
    'A': (1, 1.5),    # Example range for parameter A
    'beta': (0, 0.5)  # Example range for parameter beta
}

other_param_ranges = {
    'delta':(1e-3,1e-3)
}

# Ensure that the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Generate plots with the specified range
plot_transition_time_heatmap(results_df, focus_params, parameter_names, output_dir, 'mesoderm_lag', focus_param_ranges, other_param_ranges)