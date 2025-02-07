# -*- coding: utf-8 -*-

"""
Xylophone Key Relative Positions Generator
Group C: Zhiyu Wang, Yijia Qian, Yuan Cao

Requirements:
Library: pandas

This script generates relative positions between aruco markers and the center of the xylophone key,
creates initial and extra key points. Intial keys are the keys on the first row that will be played, the extra keys are keys 
on the second row and won't be played because NAO can't reach them. The script saves these key relative positions to a CSV file.
This script only needs to be run once if the aruco marker positions are not changed.

Developer: Yijia Qian
for the course "Humanoid Robotics System" as the final project: Task 2
"""

import numpy as np
import pandas as pd
import os
import sys
import traceback

# --------------------------------------------------------------------------------------
# Define initial relative positions for the standard keys (e.g., "Point_1")
# These relative positions are between aruco markers(11,21,31) and the xylophone keys.
# --------------------------------------------------------------------------------------
# Note: These positions are geometrically determined.
initial_relative_positions = {
    'relativepos11': [-0.085, -0.16, -0.015],
    'relativepos21': [0.07, -0.24, 0.055],
    'relativepos31': [0.055, -0.145, 0.055]
}

# --------------------------------------------------------------------------------------
# Define initial relative positions for the extra keys (e.g., "Extra_Key_1")
# --------------------------------------------------------------------------------------
# These keys might be part of an extended range on the xylophone.
initial_extra_relative_positions = {
    'relativeposup11': [-0.085 + 0.019 + 0.032 + 0.019, -0.16 + 0.0125, -0.015 + 0.008],
    'relativeposup21': [-0.07 + 0.019 + 0.032 + 0.019, -0.24 + 0.0125, 0.055 + 0.008],
    'relativeposup31': [0.055 + 0.019 + 0.032 + 0.019, -0.145 + 0.0125, 0.055 + 0.008]
}

# --------------------------------------------------------------------------------------
# Parameters for generating positions
# --------------------------------------------------------------------------------------
num_initial_points = 15        # Number of initial points (standard keys)
delta_x_initial = 0.001133     # Amount to decrement from x-axis for each standard key
delta_y_initial = 0.025        # Amount to decrement from y-axis for each standard key

num_extra_keys = 14            # Number of extra keys
delta_x_extra = 0.17 / 13      # Amount to decrement from x-axis for each extra key (≈0.013077)
delta_y_extra = 0.025          # Amount to decrement from y-axis for each extra key

# Only these extra key indices are recorded (others are skipped)
keys_to_record = [1, 3, 4, 6, 7, 8, 10, 11, 13, 14]

# --------------------------------------------------------------------------------------
# Dictionary to store relative positions for all points (both standard and extra)
# --------------------------------------------------------------------------------------
all_points = {}

# --------------------------------------------------------------------------------------
# Generate the initial 15 standard key points
# --------------------------------------------------------------------------------------
for point_index in range(num_initial_points):
    point_key = f'Point_{point_index + 1}'
    all_points[point_key] = {}
    
    for rel_key, rel_pos in initial_relative_positions.items():
        # Adjust coordinates based on the current index
        adjusted_x = rel_pos[0] - (delta_x_initial * point_index)
        adjusted_y = rel_pos[1] + (delta_y_initial * point_index)
        adjusted_z = rel_pos[2]  # z-axis remains unchanged
        
        # Store adjusted coordinates (round to 6 decimals for clarity)
        all_points[point_key][rel_key] = [
            round(adjusted_x, 6),
            round(adjusted_y, 6),
            round(adjusted_z, 6)
        ]

# --------------------------------------------------------------------------------------
# Generate extra keys' points
# --------------------------------------------------------------------------------------
for extra_key_index in range(1, num_extra_keys + 1):
    # Only proceed if this extra key is in our list to record
    if extra_key_index in keys_to_record:
        key_key = f'Extra_Key_{extra_key_index}'
        all_points[key_key] = {}
        
        for rel_key, rel_pos in initial_extra_relative_positions.items():
            # Adjust coordinates based on the current index
            adjusted_x = rel_pos[0] - (delta_x_extra * (extra_key_index - 1))
            adjusted_y = rel_pos[1] + (delta_y_extra * (extra_key_index - 1))
            adjusted_z = rel_pos[2]  # z-axis remains unchanged
            
            # Store adjusted coordinates (round to 6 decimals for clarity)
            all_points[key_key][rel_key] = [
                round(adjusted_x, 6),
                round(adjusted_y, 6),
                round(adjusted_z, 6)
            ]

# --------------------------------------------------------------------------------------
# Convert the dictionary into a DataFrame for better readability and saving
# --------------------------------------------------------------------------------------
data = []
for point, rel_positions in all_points.items():
    for rel_key, coords in rel_positions.items():
        data.append({
            'Point': point,
            'Relative Position': rel_key,
            'X': coords[0],
            'Y': coords[1],
            'Z': coords[2]
        })

df = pd.DataFrame(data)

# --------------------------------------------------------------------------------------
# Quick preview of the DataFrame
# --------------------------------------------------------------------------------------
print("DataFrame preview:")
print(df.head())

# --------------------------------------------------------------------------------------
# Print the current working directory (for debugging)
# --------------------------------------------------------------------------------------
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# --------------------------------------------------------------------------------------
# Try saving the DataFrame to a CSV file
# --------------------------------------------------------------------------------------
try:
    print("Attempting to save CSV file...")
    
    # Define the path for saving the CSV file (change as needed)
    script_directory = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
    csv_filename = 'xylophone_relative_positions.csv'
    csv_path = os.path.join(script_directory, csv_filename)
    
    # Save the DataFrame to CSV with comma separators
    df.to_csv(csv_path, index=False, sep=',')
    
    print(f"CSV file successfully saved to: {csv_path}")
except Exception as e:
    print("Error encountered while saving CSV file:", e)
    traceback.print_exc()
    sys.exit(1)
