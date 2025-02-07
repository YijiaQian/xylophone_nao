# -*- coding: utf-8 -*-

"""
Xylophone Key Position Computation for NAO Robot
Group C: Zhiyu Wang, Yijia Qian, Yuan Cao

Requirements:
Libraries: pandas, numpy, scipy


This script computes the absolute positions of xylophone keys in the NAO robot's torso frame using:
1. **Aruco Marker Localization**: Detects marker positions for reference.
2. **Relative Position Mapping**: Reads key positions from a CSV file.
3. **Least Squares Optimization**: Solves 3D key positions with respect to NAO torso frame based on spherical constraints.
4. **Arm Position Calculation**: Computes robot arm positions for key strikes.
5. **Key Mapping and Output**: Maps positions to notes and saves results to a CSV.

Developer: Yijia Qian
for the course "Humanoid Robotics System" as the final project: Task 3
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import os
import sys
import traceback
from scipy.optimize import least_squares
import math
import csv
from grasp_control import monitor_stability


# This function defines the spherical equations (constraints) for use in least_squares.
# We calculate three equations based on the distances from three known Aruco positions.
def sphere_equations(vars, aruco_position1, distance1, aruco_position2, distance2, aruco_position3, distance3):
    """
    Spherical equations for distance constraints:
    (x - x1)^2 + (y - y1)^2 + (z - z1)^2 = distance1^2
    (x - x2)^2 + (y - y2)^2 + (z - z2)^2 = distance2^2
    (x - x3)^2 + (y - y3)^2 + (z - z3)^2 = distance3^2
    """
    x, y, z = vars
    x1, y1, z1 = aruco_position1
    x2, y2, z2 = aruco_position2
    x3, y3, z3 = aruco_position3

    eq1 = (x - x1)**2 + (y - y1)**2 + (z - z1)**2 - distance1**2
    eq2 = (x - x2)**2 + (y - y2)**2 + (z - z2)**2 - distance2**2
    eq3 = (x - x3)**2 + (y - y3)**2 + (z - z3)**2 - distance3**2

    return [eq1, eq2, eq3]


# This function uses least_squares to solve the 3D coordinates that satisfy the three spherical equations.
def compute_pointpos_least_squares(
    sphere_equations,
    aruco_position1,
    distance1,
    aruco_position2,
    distance2,
    aruco_position3,
    distance3
):
    """
    Solves for the (x, y, z) that minimize the errors in the three spherical equations.
    """
    # Initial guess can be set based on prior knowledge or simply (0, 0, 0).
    initial_guess = (0.0, 0.0, 0.0)

    # Use scipy.optimize.least_squares for nonlinear least squares.
    result = least_squares(
        sphere_equations, 
        initial_guess, 
        args=(aruco_position1, distance1, aruco_position2, distance2, aruco_position3, distance3)
    )
    
    if result.success:
        print("Solved successfully: {result.x}")
        return result.x
    else:
        print("Solve failed: {result.message}")
        return None


def rotation_matrix_z(degrees):
    """rotate around z axis"""
    radians = np.deg2rad(degrees)
    cos = np.cos(radians)
    sin = np.sin(radians)
    return np.array([
        [cos, -sin, 0],
        [sin,  cos, 0],
        [0,    0,   1]
    ])

def rotation_matrix_y(degrees):
    """rotate around y axis"""
    radians = np.deg2rad(degrees)
    cos = np.cos(radians)
    sin = np.sin(radians)
    return np.array([
        [cos,  0, sin],
        [0,    1, 0],
        [-sin, 0, cos]
    ])

def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (XYZ order):
    The rotation is assumed to be R = Rz(gamma) * Ry(beta) * Rx(alpha),
    which means:
      - First rotate by alpha around X (roll)
      - Then rotate by beta around Y (pitch)
      - Finally rotate by gamma around Z (yaw)

    Parameters:
    - R: 3x3 rotation matrix

    Returns:
    - [alpha, beta, gamma]: Euler angles in radians corresponding to rotations about
      X, Y, and Z axes respectively.
    """
    # Debugging statements
    print("----- Debugging Inside rotation_matrix_to_euler_angles_XYZ -----")
    print("Type of R:", type(R))
    print("Shape of R:", R.shape)
    print("Number of dimensions (ndim):", R.ndim)
    print("R array:\n", R)

    # Ensure R is a NumPy array
    R = np.array(R)

    # Check if R is a 3x3 matrix
    if R.shape != (3, 3):
        raise ValueError("Input rotation matrix must be a 3x3 matrix.")

    # Compute beta = arcsin(R[0,2])
    beta = np.arcsin(R[0, 2])
    # Compute cos(beta) to determine if the rotation is in a singular state
    cos_beta = np.cos(beta)

    # Check for singularity (gimbal lock) using a small threshold
    if abs(cos_beta) > 1e-6:
        # Non-singular case
        alpha = np.arctan2(-R[1, 2], R[2, 2])
        gamma = np.arctan2(-R[0, 1], R[0, 0])
    else:
        # Singular case: set alpha = 0 and compute gamma from other elements
        alpha = 0
        gamma = np.arctan2(R[1, 0], R[1, 1])

    print("X (roll, alpha) (rad):", alpha)
    print("Y (pitch, beta) (rad):", beta)
    print("Z (yaw, gamma) (rad):", gamma)
    print("----- End of Debugging -----")

    return [alpha, beta, gamma]


# Main function to compute the key positions and store them in a CSV.
def compute_keyposition(results, stop_event, key_calculate_event):
    """
    Monitors results for stability, loads relative positions from CSV,
    solves for absolute positions via least squares, and writes the final arm positions to a CSV file.
    """
    try:
        # Continuously compute positions if threading Events are not set.
        while not stop_event.is_set():
            while key_calculate_event.is_set():
                # Monitor the stable results (smooth or filtered data).
                stable_results = monitor_stability(
                    results,
                    threshold=0.3,       # threshold can be tuned
                    window_size=10,      # moving window size
                    max_results=10000    # max number of results to process
                )
                results = stable_results

                print("Stable results: ", results)

                # Retrieve Aruco positions from the stable results
                for i in range(0, len(results)):
                    if results[i][0] == 11:
                        aruco_position1 = results[i][1]
                    elif results[i][0] == 21:
                        aruco_position2 = results[i][1]
                    elif results[i][0] == 31:
                        aruco_position3 = results[i][1]

                # Read the CSV file containing the relative positions
                csv_path = '/workspaces/hrs_ws/src/xylophone_relative_positions.csv'
                if not os.path.exists(csv_path):
                    print("CSV file not found: {csv_path}")
                    sys.exit(1)

                df = pd.read_csv(csv_path, sep=',')
                print("Column names:", df.columns.tolist())
                print("DataFrame preview:")
                print(df.head())

                # Acquire all unique key names (e.g. 'Point_1' or 'Extra_Key_1')
                unique_keys = df['Point'].unique()

                # Dictionary to store final solutions for each key
                solutions_dict = {}

                # Solve for the absolute position of each unique key
                for key in unique_keys:
                    key_data = df[df['Point'] == key]

                    # Decide which columns to look for based on the key name
                    if key.startswith('Extra_Key_'):
                        rel_pos_names = ['relativeposup11', 'relativeposup21', 'relativeposup31']
                    else:
                        rel_pos_names = ['relativepos11', 'relativepos21', 'relativepos31']

                    # Attempt to extract the relative positions
                    try:
                        rel_pos11 = key_data[key_data['Relative Position'] == rel_pos_names[0]][['X', 'Y', 'Z']].values[0]
                        rel_pos21 = key_data[key_data['Relative Position'] == rel_pos_names[1]][['X', 'Y', 'Z']].values[0]
                        rel_pos31 = key_data[key_data['Relative Position'] == rel_pos_names[2]][['X', 'Y', 'Z']].values[0]
                    except IndexError:
                        print("Missing necessary relative position data for {key}, skipping...")
                        continue

                    # Compute Euclidean distances
                    distance1 = np.linalg.norm(rel_pos11)
                    distance2 = np.linalg.norm(rel_pos21)
                    distance3 = np.linalg.norm(rel_pos31)

                    # Solve using least squares
                    solution = compute_pointpos_least_squares(
                        sphere_equations,
                        aruco_position1, distance1,
                        aruco_position2, distance2,
                        aruco_position3, distance3
                    )

                    # Store solution in a dictionary if successful
                    if solution is not None:
                        solutions_dict[key] = solution.tolist()
                    else:
                        solutions_dict[key] = None

                # Print all solutions
                print("\nAll solutions:")
                for key, value in solutions_dict.items():
                    print("{key}: {value}")

                # Key mapping for xylophone notes
                key_mapping = {
                    "Point_1": "G7",
                    "Point_2": "F7",
                    "Point_3": "E7",
                    "Point_4": "D7",
                    "Point_5": "C7",
                    "Point_6": "B6",
                    "Point_7": "A6",
                    "Point_8": "G6",
                    "Point_9": "F6",
                    "Point_10": "E6",
                    "Point_11": "D6",
                    "Point_12": "C6",
                    "Point_13": "B5",
                    "Point_14": "A5",
                    "Point_15": "G5",
                }

                # These parameters are fixed in our scenario
                length = 0.188  # Link length in centimeters
                # Prepare to store results
                result_rows = []
                # Calculate the arm start position for each solution
                for i, (point_key, value) in enumerate(solutions_dict.items(), start=1):
                    if i == 16:
                        continue

                    x_e, y_e, z_e = value
                    # Determine which hand to use, based on the count (1-7 => hand 1, 8-15 => hand 2)
                    hand = 1 if i <= 7 else 2
                    wrist_angle = -1.57 if hand == 1 else 1.57
                    mapped_key = key_mapping[point_key]
                    
                    # define the rotation matrices
                    Rz1 = [rotation_matrix_z(90) if hand == 1 else rotation_matrix_z(-90)][0]
                    Rz2 = [rotation_matrix_z(-20) if hand == 1 else rotation_matrix_z(20)][0]
                    Ry = rotation_matrix_y(30)
                    Rz3 = rotation_matrix_z(90)
                    stick_direction = np.array([-length, 0, 0])
                    # calculate the final rotation matrix of the arm with respect to the torso
                    R_total = np.dot(np.dot(np.dot(Rz1, Rz2), Ry), Rz3)
                    print("Type of R_total:", type(R_total))
                    print("Shape of R_total:", np.shape(R_total))
                    print("R_total array:\n", R_total)
                    # calculate the final position of the arm with respect to the torso
                    [x_s, y_s, z_s] = np.dot(np.dot(np.dot(Rz1, Rz2), Ry),stick_direction) + np.array([x_e, y_e, z_e])
                    euler_angles = rotation_matrix_to_euler_angles(R_total)
                    result_rows.append({
                        "count": i,
                        "key": mapped_key,
                        "hand": hand,
                        "arm_position": [x_s, y_s, z_s] + euler_angles,
                        "strike_wrist_angle": [wrist_angle],
                        "time": 0.25                
                    })

                # Write the final results to a CSV file
                output_file = "keyposition_copy.csv"
                fieldnames = ["count", "key", "hand", "arm_position", "strike_wrist_angle", "time"]
                
                with open(output_file, 'wb') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(result_rows)

                print("Results have been saved to {output_file}")
                key_calculate_event.clear()
            pass

    except Exception as e:
        print("Error occurred:", e)
        traceback.print_exc()
        sys.exit(1)
    
    except KeyboardInterrupt:
        sys.exit(0)
