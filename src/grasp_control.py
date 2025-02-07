# -*- coding: utf-8 -*-

"""
NAO Robot Hand Control for Grasping the Mallets (Sticks for Xylophone)
Group C: Zhiyu Wang, Yijia Qian, Yuan Cao

Requirements: 
NAOqi API Version: 2.1
Libraries: numpy, scipy, opencv

This script contains the functions for calculating the grasp positions and executing the grasp sequence.
For the stable results fetched from the ArUco marker detection, a support sector, monitor_stability, is used to monitor
the stability of the marker.
After the stability is confirmed, the average of the stable results is calculated to minimize the noise.
Then, the grasp positions are calculated for both hands, and the trajectory is planned for the approach, grasp,
and lift positions.
Transfer the calculated grasp positions to the path_queue for the execution of the grasp sequence as another thread.
Both threads run concurrently until the stop_event has been set, which is triggered by the main thread.
With the grasp_event and grasp_action, the grasp sequence is executed.

The robot will detect the stiffness (by motionProxy.getStiffnesses) of both hands and provide feedback to the user
through the text-to-speech module.
Once the stiffness of both hands is set to 1.0, the grasp is successful, the grasp_action is cleared.
While not, the grasp_event is cleared, and the get_path_event is set to request a new path.

!!!IMPORTANT NOTICE!!!:
1.  The getStiffnesses is not stable and may not work as expected.
2.  The grasp orientation may not working as expected, due to Cartesian coordinates and Euler angles' inconsistency
    (or other possible reasons).
   

Developer: Yuan Cao, Zhiyu Wang
for the course "Humanoid Robotics System" as the final project: Task 2
"""
from __future__ import print_function

import time
import numpy as np
import multiprocessing
import cv2

from collections import deque, defaultdict
from scipy.spatial.transform import Rotation as R

def calculate_grasp_positions(data_queue, path_queue, stop_event, get_path_event, grasp_action):
    from naoqi import ALProxy
    tts = ALProxy("ALTextToSpeech", "10.152.246.194", 9559)
    try:
        while not stop_event.is_set():
            time.sleep(2)
            while get_path_event.is_set() and grasp_action.is_set():
                
                tts.say("I am calculating new grasp position.")
                
                got_marker_41 = False
                while got_marker_41 == False:
                    """Calculate the three positions for approach, grasp, and lift"""
                    
                    stable_results = monitor_stability(
                        data_queue,             # queue to consume marker results
                        threshold=0.6,          # set a threshold for stability
                        window_size=30,         # windows size for stability check
                        max_results=10000       # maximum number of results to process to avoid
                    )
                    for i in range(len(stable_results)):
                        if stable_results[i][0] == 41:
                            got_marker_41 = True
                            break
                    
                    
                hand_list = []
                trajectory = []
                
                # Transform Martrix for Both Hands, from aruco marker frame to torso frame, 
                t_aruco_torso = np.array(stable_results[0][1])
                T_aruco_torso = np.eye(4)
                
                T_aruco_torso[:3, 3] = t_aruco_torso
                
                # Use Rodrigues to convert the rotation vector to rotation matrix
                rotvector_aruco_torso = np.array(stable_results[0][2])
                r_aruco_torso, _ = cv2.Rodrigues(rotvector_aruco_torso)
                T_aruco_torso[:3, :3] = r_aruco_torso
                
                for i in range(len(stable_results)):
                    if stable_results[i][0] == 41:
                        """ 
                        Right Hand Grasp Position Calculation 
                        """
                        trajectory_r = []
                        handname = "RArm"
                        hand_list.append(handname)
                        
                        # End effector position and orientation in Aruco Marker frame
                        t_rarm_aruco = [0.07, 0.01, 0.04]
                        
                        # Euler angles: XYZ
                        roll_x_rarm  = np.deg2rad(175)          # Roll  : X
                        pitch_y_rarm = np.deg2rad(72)           # Pitch : Y
                        yaw_z_rarm   = np.deg2rad(170)          # Yaw   : Z
                        
                        euler_rarm_aruco = [roll_x_rarm, pitch_y_rarm, yaw_z_rarm]

                        rot_rarm_aruco = R.from_euler('xyz', euler_rarm_aruco)
                        r_rarm_aruco = rot_rarm_aruco.as_dcm()
                        
                        T_rarm_aruco = np.eye(4)
                        T_rarm_aruco[:3, :3] = r_rarm_aruco
                        T_rarm_aruco[:3, 3] = t_rarm_aruco
                        
                        # End effector position and orientation in Torso frame
                        T_rarm_torso = np.dot(T_aruco_torso, T_rarm_aruco)
                        
                        # Extract position and Euler angles XYZ from the transformation matrix
                        t_rarm_torso = T_rarm_torso[:3, 3]
                        r_rarm_torso = T_rarm_torso[:3, :3]
                        rot_rarm_torso = R.from_dcm(r_rarm_torso)
                        euler_rarm_torso = rot_rarm_torso.as_euler('xyz').tolist()
                    
                        # print("Rarm pose:",euler_rarm_torso)
                        
                        approach_pos = []
                        grasp_pos = []
                        lift_pos = []
                        
                        grasp_orientation_r = euler_rarm_torso
                        
                        # Approach position: offset from marker
                        approach_pos.append(t_rarm_torso[0]-0.04)
                        approach_pos.append(t_rarm_torso[1]-0.04)
                        approach_pos.append(t_rarm_torso[2])
                        approach_pose_r = approach_pos + grasp_orientation_r
                        trajectory_r.append(approach_pose_r)
                        
                        # Grasp position
                        grasp_pos.append(t_rarm_torso[0])
                        grasp_pos.append(t_rarm_torso[1])
                        grasp_pos.append(t_rarm_torso[2])
                        grasp_pose_r = grasp_pos + grasp_orientation_r
                        trajectory_r.append(grasp_pose_r)

                        # Lift position: higher Z
                        lift_pos.append(t_rarm_torso[0])
                        lift_pos.append(t_rarm_torso[1])
                        lift_pos.append(t_rarm_torso[2]+0.08)
                        lift_pose_r = lift_pos + grasp_orientation_r
                        trajectory_r.append(lift_pose_r)
                        
                        trajectory.append(trajectory_r)
                        
                        """ 
                        Left Hand Grasp Position Calculation 
                        """
                        trajectory_l = []
                        handname = "LArm"
                        hand_list.append(handname)
                        
                        # End effector position and orientation in Aruco Marker frame
                        t_larm_aruco = [-0.07, 0.02, 0.06]

                        # Euler angles: XYZ
                        roll_x_larm  = np.deg2rad(-135)             # Roll  : X
                        pitch_y_larm = np.deg2rad(69)               # Pitch : Y
                        yaw_z_larm   = np.deg2rad(56)               # Yaw   : Z
                        
                        euler_larm_aruco = [roll_x_larm, pitch_y_larm, yaw_z_larm]

                        rot_larm_aruco = R.from_euler('xyz', euler_larm_aruco)
                        r_larm_aruco = rot_larm_aruco.as_dcm()

                        T_larm_aruco = np.eye(4)
                        T_larm_aruco[:3, :3] = r_larm_aruco
                        T_larm_aruco[:3, 3] = t_larm_aruco
                        
                        # End effector position and orientation in Torso frame
                        T_larm_torso = np.dot(T_aruco_torso, T_larm_aruco)
                        
                        # Extract position and Euler angles ZYX from the transformation matrix
                        t_larm_torso = T_larm_torso[:3, 3]
                        r_larm_torso = T_larm_torso[:3, :3]
                        rot_larm_torso = R.from_dcm(r_larm_torso)
                        euler_larm_torso = rot_larm_torso.as_euler('xyz').tolist()
                        
                        # print("Larm pose:",euler_torso_larm)
                        
                        approach_pos = []
                        grasp_pos = []
                        lift_pos = []

                        grasp_orientation_l = euler_larm_torso
                        
                        # Approach position: offset from marker
                        approach_pos.append(t_larm_torso[0]-0.04)
                        approach_pos.append(t_larm_torso[1]+0.04)
                        approach_pos.append(t_larm_torso[2])
                        approach_pose_l = approach_pos + grasp_orientation_l
                        trajectory_l.append(approach_pose_l)
                        
                        # Grasp position: closer to marker
                        grasp_pos.append(t_larm_torso[0])
                        grasp_pos.append(t_larm_torso[1])
                        grasp_pos.append(t_larm_torso[2])
                        grasp_pose_l = grasp_pos + grasp_orientation_l
                        trajectory_l.append(grasp_pose_l)

                        # Lift position: higher Z
                        lift_pos.append(t_larm_torso[0])
                        lift_pos.append(t_larm_torso[1])
                        lift_pos.append(t_larm_torso[2]+0.08)
                        lift_pose_l = lift_pos + grasp_orientation_l
                        trajectory_l.append(lift_pose_l)
                        
                        trajectory.append(trajectory_l)
                
                # transfer the grasp position to the path_queue
                path_transfer=[]
                path_transfer.append(hand_list)
                path_transfer.append(trajectory)
                if path_queue.full():
                    path_queue.get()
                path_queue.put(path_transfer)
                        
                
                time.sleep(3)
                pass
            pass
    except KeyboardInterrupt:
        pass
    finally:
        print("Grasp position calculation stopped.")
              


def execute_grasp_sequence(robot_ip, robot_port, path_queue, stop_event, get_path_event, grasp_event, grasp_action):
    from naoqi import ALProxy
    import motion
    import time
    
    motionProxy = ALProxy("ALMotion", robot_ip, robot_port)
    tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
    attempt = 0
    motionProxy.setStiffnesses("RHand", 0.0)
    motionProxy.setStiffnesses("LHand", 0.0)
    
    try:
        """Execute the complete grasp sequence"""
        while not stop_event.is_set():
            while path_queue.empty():
                get_path_event.set()
            if not path_queue.empty():
                arm_name = []
                path_list = []
                arm_name, path_list = path_queue.get()
                get_path_event.clear()
                pass
            
            while not grasp_event.is_set() and grasp_action.is_set() and not stop_event.is_set() and not get_path_event.is_set():
                for i in range(len(arm_name)):
                    if arm_name[i] == "RArm":
                        hand_name = "RHand"
                        hand_name_tts = "Right Hand"
                    elif arm_name[i] == "LArm":
                        hand_name = "LHand"
                        hand_name_tts = "Left Hand"

                    tts_approach = "I am moving my " + hand_name_tts + " to grasp the Stick." 
                    tts_grasped = "My " +  hand_name_tts + " has grasped the Stick!" 

                    # 1. Move arm to approach position
                    tts.say(tts_approach)
                    motionProxy.positionInterpolations(
                        arm_name[i], motion.FRAME_TORSO,
                        path_list[i][0], [motion.AXIS_MASK_ALL], [2.0]
                    )

                    # 2. Open hand
                    motionProxy.openHand(hand_name)

                    # 3. Move arm to grasp position
                    motionProxy.positionInterpolations(
                        arm_name[i], motion.FRAME_TORSO,
                        path_list[i][1], [motion.AXIS_MASK_ALL], [2.0]
                    )

                    # 4. Close hand and stiffen
                    motionProxy.closeHand(hand_name)
                    motionProxy.setStiffnesses(hand_name, 1.0)
                    time.sleep(0.5)
                    tts.say(tts_grasped)
                    
                    # 5. Lift position
                    motionProxy.positionInterpolations(
                        arm_name[i], motion.FRAME_TORSO,
                        path_list[i][2], [motion.AXIS_MASK_ALL], [2.0]
                    )
                    
                # Check the stiffness of both hands
                lhand_stiffnesses = motionProxy.getStiffnesses("LHand")
                rhand_stiffnesses = motionProxy.getStiffnesses("RHand")
                check_stiffness_tts = "Right Hand Stiffness is " + str(rhand_stiffnesses) + ", Left Hand Stiffness is " + str(lhand_stiffnesses)
                tts.say(check_stiffness_tts)
                
                if lhand_stiffnesses == [1.0] and rhand_stiffnesses == [1.0]:
                    grasp_action.clear()
                    grasp_event.set()
                    tts.say("Both hands have grasped the Stick!")
                    attempt = 0
                    break
                
                if lhand_stiffnesses == [0.0] or rhand_stiffnesses == [0.0]:
                    if attempt< 2:                  # set the maximum attempt to 3
                        grasp_event.clear()
                        get_path_event.set() # request for new path
                        tts.say("Failed to grasp the Stick. I will try again")
                        attempt += 1
                        break
                    else:
                        grasp_action.clear()
                        grasp_event.clear()
                        tts.say("Failed to grasp the Stick. I will stop the grasp action.")
                        attempt = 0
                        break
                
    finally:
        print("Grasp sequence execution stopped.")
        pass
        


"""
Result parsing and stability monitoring functions
"""   
def parse_result(result):
    markers_data = {}
    try:
        if not isinstance(result, list):
            raise ValueError("The result is not a List")
        if not result:
            # print("No markers detected.")
            return markers_data  # Return empty dictionary

        for marker_entry in result:
            if not isinstance(marker_entry, list) or len(marker_entry) < 1:
                print("A marker in the result has an incorrect format. Skipping.")
                continue
            marker_id, values1, values2 = marker_entry
            if not (isinstance(values1, list) and isinstance(values2, list)):
                print("Marker ID {marker_id} has an incorrect sublist format. Skipping.")
                continue
            all_values = values1 + values2
            markers_data[marker_id] = all_values
        return markers_data
    except Exception as e:
        print("Error in exception:", e)
        return markers_data  # Return the part that has been parsed so far

def compute_max_min_diff(results_window):
    if not results_window:
        return []

    num_params = len(results_window[0])
    max_min_diffs = [0.0] * num_params

    for i in range(num_params):
        param_values = [result[i] for result in results_window]
        max_val = max(param_values)
        min_val = min(param_values)
        max_min_diffs[i] = max_val - min_val

    return max_min_diffs

def is_stable(max_min_diffs, threshold):
    """
    Determines if all parameters' max-min differences are less than or equal to the threshold.
    """
    for diff in max_min_diffs:
        if diff > threshold:
            return False
    return True


def monitor_stability(q, threshold=0.001, window_size=20, max_results=1000):
    """
    Monitors the stability of markers by collecting 20 stable frames for each marker,
    computing their average, and returning all the final results as a list.

    Args:
        q (Queue): Queue from which marker results are consumed.
        threshold (float): Threshold for stability.
        window_size (int): Number of recent results to consider for stability.
        max_results (int): Maximum number of results to process.

    Returns:
        list: A list of [marker_id, tvec, rvec] for each stable marker.
    """
    markers_windows = {}          # key: marker_id, value: deque of last 'window_size' results
    stable_count = defaultdict(int)  # Count of stable instances per marker
    final_results = []               # List to store [marker_id, tvec, rvec]
    processed = 0
    max_parse_failures = 10
    parse_failures = 0

    while processed < max_results:
        try:
            result = q.get(timeout=1)  # Set timeout to avoid infinite blocking
        except Exception:
            print("Waiting for results timed out. The producer might have stopped.")
            return final_results

        if result is None:
            print("Producer has ended.")
            return final_results

        markers_data = parse_result(result)

        # If no markers were parsed, continue
        if not markers_data:
            parse_failures += 1
            if parse_failures >= max_parse_failures:
                print("Reached maximum parse failure count. Terminating monitoring.")
                break
            continue
        parse_failures = 0  # Reset parse failure count on successful parse

        # Process and store each marker's result
        for marker_id, current_values in markers_data.items():
            if marker_id not in markers_windows:
                markers_windows[marker_id] = deque(maxlen=window_size)

            # Append current result to the marker's window
            markers_windows[marker_id].append(current_values)

            # When the window is full, check for stability
            if len(markers_windows[marker_id]) == window_size:
                diffs = compute_max_min_diff(markers_windows[marker_id])
                if is_stable(diffs, threshold):
                    # If stable, increment the stable count
                    stable_count[marker_id] += 1

                    # When a marker has 20 stable counts, compute the average and add to final_results
                    if stable_count[marker_id] == 20:
                        # Calculate the average of the 20 stable results
                        window_list = list(markers_windows[marker_id])
                        param_count = len(window_list[0])
                        avg_values = []
                        for i in range(param_count):
                            avg = sum(entry[i] for entry in window_list) / float(window_size)
                            avg_values.append(avg)
                        # Split avg_values into tvec and rvec
                        tvec = avg_values[:3]  # Assuming first 3 values are tvec
                        rvec = avg_values[3:]  # Assuming next 3 values are rvec
                        # Append to final_results as a sublist
                        final_results.append([marker_id, tvec, rvec])
                        print ("Marker ID: {marker_id} has reached 20 stable results.".format(marker_id=marker_id))
                else:
                    # If not stable, reset the stable count
                    stable_count[marker_id] = 0

        processed += 1

        # Check if all known markers have reached 20 stable results
        # This condition assumes that once a marker is in final_results, it's done
        if markers_windows and all(
            any(fr[0] == marker_id for fr in final_results) for marker_id in markers_windows
        ):
            print("All known markers have collected 20 stable results.")
            return final_results

    # After processing all results or reaching max_results, return the final_results
    if final_results:
        return final_results
    else:
        print("No markers achieved 20 stable results. Returning the last available results.")
        last_results = []
        for marker_id, window in markers_windows.items():
            if window:
                avg_values = window[-1]
                tvec = avg_values[:3]
                rvec = avg_values[3:]
                last_results.append([marker_id, tvec, rvec])
        return last_results