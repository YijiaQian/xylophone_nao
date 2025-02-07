#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
ArUco Marker to NAO Robot Arm Transformation Measurement
Group C: Zhiyu Wang, Yijia Qian, Yuan Cao

Requirements:
Naoqi Python SDK Version: 2.1
Libraries: numpy, cv2, scipy

This script connects to NAO's camera and continuously detects ArUco markers
using the ArucoDetector. The detected markers are transformed into the torso frame.
By loosening each arm, positioning it near the marker, and reading its transform
via ALMotion, we compute T_rarm->marker or T_larm->marker. The final 4×4 transform
matrix and corresponding Euler angles (XYZ convention) are printed to the console.

Outputs:
- Transformation matrix: T_arm->marker
- Euler angles in the XYZ convention


Developer: Yuan Cao
for the course "Humanoid Robotics System" as the final project (Task 2)
"""

from __future__ import print_function
import time
import cv2
import threading
import Queue
import numpy as np
from scipy.spatial.transform import Rotation as R
from naoqi import ALProxy, ALBroker
import motion

from aruco_marker import ArucoDetector
from nao_service import robot_init


# Camera parameters for ArUco detection
camerapara_dict = {
    0: np.array([]),
    1: np.array([
        [286.6866057, 0.000000, 162.2270394],
        [0.000000, 285.2998696, 105.4745985],
        [0.000000, 0.000000, 1.000000]
    ], dtype=np.float32),
    2: np.array([
        [562.3129566280151, 0.000000, 324.21437038284097],
        [0.000000, 555.8965725951891, 217.4121628010703],
        [0.000000, 0.000000, 1.000000]
    ], dtype=np.float32),
    3: np.array([
        [1147.444868, 0.000000, 644.3633676],
        [0.000000, 1143.986342, 430.8996752],
        [0.000000, 0.000000, 1.000000]
    ], dtype=np.float32)
}

cameradist = np.array([
    -0.06665819242416217,  0.09060075882427537,
    -0.00012550218643474006, -0.0012131476680471336,
    -0.05834098541272104
], dtype=np.float32)

def capture_video(robot_ip, robot_port, data_queue, camera_id, resolution, stop_event):
    """
    Continuously captures frames from NAO's camera and detects ArUco markers.
    Pushes detection results into data_queue as:
        [
            [marker_id, [x, y, z], [rx, ry, rz]],
            ...
        ]
    where each marker pose is in the torso frame.
    Exits when 'stop_event' is set or 'q' is pressed.
    """
    video_proxy = ALProxy("ALVideoDevice", robot_ip, robot_port)
    name_id = "Measure_EE_to_ArUco_Marker"
    fps = 20
    color_space = 13  # BGR
    video_client = video_proxy.subscribeCamera(name_id, camera_id, resolution, color_space, fps)
    
    detector = ArucoDetector(camerapara_dict[resolution], cameradist)

    try:
        while not stop_event.is_set():
            nao_image = video_proxy.getImageRemote(video_client)
            if nao_image is None:
                print("No image received from camera.")
                continue

            width = nao_image[0]
            height = nao_image[1]
            array = nao_image[6]
            img = np.frombuffer(bytearray(array), dtype=np.uint8).reshape((height, width, 3))

            results, processed_img = detector.detect_and_transform(img)

            if data_queue.full():
                data_queue.get()
            data_queue.put(results)

            cv2.imshow("NAO Video Stream", processed_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    finally:
        video_proxy.unsubscribe(video_client)
        cv2.destroyAllWindows()
        print("Video streaming stopped.")

def measure_rarm_to_marker(robot_ip, robot_port, data_queue, marker_id=41):
    """
    1) Loosen the right arm.
    2) Wait for user input to confirm the arm is near the marker.
    3) Take the last known detection of 'marker_id' from data_queue (in torso frame).
    4) Get T_rarm->torso from ALMotion.
    5) Build T_marker->torso from the ArUco detection (Rodrigues + translation).
    6) Compute T_rarm->marker = inv(T_marker->torso) * T_rarm->torso.
    7) Convert to Euler angles (XYZ) and print the result.
    """
    motion_proxy = ALProxy("ALMotion", robot_ip, robot_port)
    tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)

    tts.say("I am loosening my right arm. Please move my hand near marker {}.".format(marker_id))
    motion_proxy.setStiffnesses("RArm", 0.0)
    motion_proxy.setStiffnesses("RHand", 0.0)

    raw_input("Press ENTER when the right hand is positioned near marker {}...".format(marker_id))

    latest_marker_pose = None
    while not data_queue.empty():
        result_batch = data_queue.get()
        for (m_id, tvec, rvec) in result_batch:
            if m_id == marker_id:
                latest_marker_pose = (tvec, rvec)

    if latest_marker_pose is None:
        tts.say("Marker {} not detected recently. Please try again.".format(marker_id))
        return

    (marker_tvec, marker_rvec) = latest_marker_pose
    T_rarm_torso_list = motion_proxy.getTransform("RArm", motion.FRAME_TORSO, True)
    T_rarm_torso = np.array(T_rarm_torso_list).reshape(4, 4)

    R_marker_torso, _ = cv2.Rodrigues(np.array(marker_rvec))
    T_marker_torso = np.eye(4)
    T_marker_torso[:3, :3] = R_marker_torso
    T_marker_torso[:3, 3] = np.array(marker_tvec)

    T_marker_torso_inv = np.linalg.inv(T_marker_torso)
    T_rarm_marker = np.dot(T_marker_torso_inv, T_rarm_torso)

    R_rarm_marker = T_rarm_marker[:3, :3]
    rot_rarm = R.from_matrix(R_rarm_marker).as_euler('xyz', degrees=True)

    tts.say("Measurement complete. See console.")
    print("\n==== T_rarm->marker  (Marker ID:{}) ====".format(marker_id))
    print(T_rarm_marker)
    print("Rotation (3×3):\n", R_rarm_marker)
    print("Translation (XYZ):\n", T_rarm_marker[:3, 3])
    print("Rotation in Euler (XYZ) =", rot_rarm)
    print("=========================================\n")

def measure_larm_to_marker(robot_ip, robot_port, data_queue, marker_id=41):
    """
    1) Loosen the left arm.
    2) Wait for user input to confirm the arm is near the marker.
    3) Take the last known detection of 'marker_id' from data_queue (in torso frame).
    4) Get T_larm->torso from ALMotion.
    5) Build T_marker->torso from the ArUco detection (Rodrigues + translation).
    6) Compute T_larm->marker = inv(T_marker->torso) * T_larm->torso.
    7) Convert to Euler angles (XYZ) and print the result.
    """
    motion_proxy = ALProxy("ALMotion", robot_ip, robot_port)
    tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)

    tts.say("I am loosening my left arm. Please move my hand near marker {}.".format(marker_id))
    motion_proxy.setStiffnesses("LArm", 0.0)
    motion_proxy.setStiffnesses("LHand", 0.0)

    raw_input("Press ENTER when the left hand is positioned near marker {}...".format(marker_id))

    latest_marker_pose = None
    while not data_queue.empty():
        result_batch = data_queue.get()
        for (m_id, tvec, rvec) in result_batch:
            if m_id == marker_id:
                latest_marker_pose = (tvec, rvec)

    if latest_marker_pose is None:
        tts.say("Marker {} not detected recently. Please try again.".format(marker_id))
        return

    (marker_tvec, marker_rvec) = latest_marker_pose
    T_larm_torso_list = motion_proxy.getTransform("LArm", motion.FRAME_TORSO, True)
    T_larm_torso = np.array(T_larm_torso_list).reshape(4, 4)

    L_marker_torso, _ = cv2.Rodrigues(np.array(marker_rvec))
    T_marker_torso = np.eye(4)
    T_marker_torso[:3, :3] = L_marker_torso
    T_marker_torso[:3, 3] = np.array(marker_tvec)

    T_marker_torso_inv = np.linalg.inv(T_marker_torso)
    T_larm_marker = np.dot(T_marker_torso_inv, T_larm_torso)

    R_larm_marker = T_larm_marker[:3, :3]
    rot_larm = R.from_matrix(R_larm_marker).as_euler('xyz', degrees=True)

    tts.say("Measurement complete. See console.")
    print("\n==== T_larm->marker  (Marker ID:{}) ====".format(marker_id))
    print(T_larm_marker)
    print("Rotation (3×3):\n", R_larm_marker)
    print("Translation (XYZ):\n", T_larm_marker[:3, 3])
    print("Rotation in Euler (XYZ) =", rot_larm)
    print("=========================================\n")

def main(robot_ip="10.152.246.194", robot_port=9559):
    """
    Initializes the NAO robot, starts the video capture thread, and provides a command-line
    interface to measure rArm->marker or lArm->marker. Quits on user request.
    """
    asrBroker = ALBroker("asrBroker", "0.0.0.0", 0, robot_ip, robot_port)
    tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)

    stop_event = threading.Event()
    data_queue = Queue.Queue()

    robot_init_thread = threading.Thread(target=robot_init, args=(robot_ip, robot_port, stop_event))
    robot_init_thread.start()
    time.sleep(5)

    camera_id = 1
    resolution = 2
    video_thread = threading.Thread(target=capture_video,
                                    args=(robot_ip, robot_port, data_queue, camera_id, resolution, stop_event))
    video_thread.start()
    time.sleep(2)

    tts.say("Minimal client is ready. No automatic events are triggered.")

    try:
        while True:
            cmd = raw_input("\nType 'r' to measure RArm->marker, 'l' to measure LArm->marker,  or 'q' to quit: ").strip()
            if cmd.lower() == 'q':
                stop_event.set()
                break
            elif cmd.lower() == 'r':
                measure_rarm_to_marker(robot_ip, robot_port, data_queue, marker_id=41)
            elif cmd.lower() == 'l':
                measure_larm_to_marker(robot_ip, robot_port, data_queue, marker_id=41)
            else:
                print("Unknown command:", cmd)
    except KeyboardInterrupt:
        stop_event.set()

    robot_init_thread.join()
    video_thread.join()
    print("Program stopped.")

if __name__ == "__main__":
    robot_ip = "10.152.246.194"
    robot_port = 9559
    print("Connecting to NAO robot at IP {} on port {}.".format(robot_ip, robot_port))
    main(robot_ip, robot_port)
