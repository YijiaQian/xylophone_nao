"""
Aruco Marker Detection
Group C: Zhiyu Wang, Yijia Qian, Yuan Cao

Requirements:
NAOqi API Version: 2.1
Libraries: numpy, cv2

This module provides a class for detecting ArUco markers in images, estimating their poses,
and averaging the results over a time window. The class also overlays the detection information
on the image and displays the averaged results in text form.
The results are stored in a buffer and updated in a thread-safe manner. Other modules can access
the averaged results for further processing.

Developer: Zhiyu Wang, Yuan Cao
for the course "Humanoid Robotics System" as the final project
"""

from __future__ import print_function

import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time
import threading

import motion
from naoqi import ALProxy

# Robot connection details
robot_ip = "10.152.246.194"
robot_port = 9559
motion_proxy = ALProxy("ALMotion", robot_ip, robot_port)

class ArucoDetector:
    def __init__(self, camera_matrix, dist_coeffs, averaging_window=1.0):
        """
        :param camera_matrix: The camera matrix for pose estimation, provided by calibration.
        :param dist_coeffs: Distortion coefficients of the camera, provided by calibration.
        :param averaging_window: Time window (in seconds) to average detections (default 1.0s).

        detect_id: List of ArUco marker IDs to detect [11, 21, 31, 41]. 
                   Others will be ignored to avoid noise.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.detect_id = [11, 21, 31, 41]
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_1000)     # ArUco dictionary: 5x5_1000
        self.marker_size = 0.06                                         # Marker size in meters
        self.aruco_params = aruco.DetectorParameters_create()

        # Buffer to store detections for stability: marker_id -> list of (timestamp, tvec, rvec)
        self.buffer = {marker_id: [] for marker_id in self.detect_id}
        self.averaging_window = averaging_window                        # 1 second by default

        # Buffer to store current frame detections for display: list of [marker_id, tvec, rvec]
        self.display_detections = []

        # This will hold the time-averaged pose for each marker each frame
        self.current_average = []

        # Thread lock for safe buffer access
        self.lock = threading.Lock()

    def detect_and_transform(self, cv_image):
        """
        Detects ArUco markers in the given image, estimates their poses, updates the buffers,
        computes averaged poses, and overlays the information on the image.

        :param cv_image: The input image in which to detect markers.
        :return: A tuple containing:
                 - current_average (list): [ [marker_id, avg_tvec, avg_rvec], ... ] over the last 1s
                 - display_detections (list): Markers detected in the current frame [id, tvec, rvec]
                 - processed_image (numpy.ndarray): The image with overlays.
        """
        current_time = time.time()
        image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in the image
        corners_detected, ids_detected, rejected = aruco.detectMarkers(
            image_gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        # Initialize display detections for this frame
        self.display_detections = []

        # Filter out the detected markers we actually care about
        detected_ids_current_frame = set()
        corners = []
        ids = []
        if ids_detected is not None:
            for i in range(len(ids_detected)):
                marker_id = int(ids_detected[i][0])
                if marker_id in self.detect_id:
                    corners.append(corners_detected[i])
                    ids.append(marker_id)
                    detected_ids_current_frame.add(marker_id)

        # Draw bounding boxes for the current frame
        if len(corners) > 0:
            for (marker_corners, marker_id) in zip(corners, ids):
                # Reshape marker corners
                c = marker_corners.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = c

                top_left = (int(top_left[0]), int(top_left[1]))
                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

                # Draw the bounding box
                cv2.line(cv_image, top_left, top_right, (0, 255, 0), 2)
                cv2.line(cv_image, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(cv_image, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(cv_image, bottom_left, top_left, (0, 255, 0), 2)

                # Calculate and draw center
                center_x = int((top_left[0] + bottom_right[0]) / 2.0)
                center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                cv2.circle(cv_image, (center_x, center_y), 4, (0, 0, 255), -1)

                # Draw the ArUco ID
                cv2.putText(
                    cv_image,
                    str(marker_id),
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (18, 226, 198),
                    1
                )

        # Estimate the pose of each detected marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_size,
            self.camera_matrix,
            self.dist_coeffs
        )

        # Get transform from optical frame to torso frame
        T_camera_to_torso = np.array(
            motion_proxy.getTransform("CameraBottom", motion.FRAME_TORSO, True)
        ).reshape(4, 4)
        T_optical_to_camera = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
        T_optical_to_torso = np.dot(T_camera_to_torso, T_optical_to_camera)

        # Lock the buffer while updating
        with self.lock:
            try:
                for i, marker_id in enumerate(ids):
                    # Draw the axes for visual debugging
                    aruco.drawAxis(
                        cv_image,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvecs[i],
                        tvecs[i],
                        0.025
                    )

                    # Construct full transform for marker in camera frame
                    rot_matrix_optical, _ = cv2.Rodrigues(rvecs[i])
                    T_marker_to_optical = np.eye(4)
                    T_marker_to_optical[:3, :3] = rot_matrix_optical
                    T_marker_to_optical[:3, 3] = tvecs[i].flatten()

                    T_marker_to_torso = np.dot(T_optical_to_torso, T_marker_to_optical)
                    tvec_marker_to_torso = T_marker_to_torso[:3, 3].tolist()
                    rvec_marker_to_torso = cv2.Rodrigues(T_marker_to_torso[:3, :3])[0].flatten().tolist()

                    # -- FIX: Re-create the buffer list if missing --
                    if marker_id not in self.buffer:
                        self.buffer[marker_id] = []
                    
                    self.buffer[marker_id].append(
                        (current_time, tvec_marker_to_torso, rvec_marker_to_torso)
                    )

                    # Also store the detection for immediate display
                    self.display_detections.append([
                        marker_id, tvec_marker_to_torso, rvec_marker_to_torso
                    ])

                # Now remove old entries from each marker's buffer
                for marker_id in list(self.buffer.keys()):
                    self.buffer[marker_id] = [
                        entry for entry in self.buffer[marker_id]
                        if current_time - entry[0] <= self.averaging_window
                    ]
                    # If empty, delete from dictionary
                    if not self.buffer[marker_id]:
                        del self.buffer[marker_id]

                # Compute averages for each marker still in the buffer
                averaged_results = []
                for marker_id, entries in self.buffer.items():
                    if entries:
                        avg_tvec = np.mean([entry[1] for entry in entries], axis=0).tolist()
                        avg_rvec = np.mean([entry[2] for entry in entries], axis=0).tolist()
                        averaged_results.append([marker_id, avg_tvec, avg_rvec])
                self.current_average = averaged_results

                # Prepare extra white space at the bottom for text display
                height, width, channels = cv_image.shape
                padding_height = 220  # Enough space for multiple lines
                new_height = height + padding_height

                expanded_image = np.zeros((new_height, width, 3), dtype=np.uint8)
                expanded_image[:height, :] = cv_image
                expanded_image[height:, :] = (255, 255, 255)

                # Display averaged results (marker_id, pos, rot)
                y_offset = height + 20
                for result in self.current_average:
                    marker_id, avg_tvec, avg_rvec = result
                    # Convert rotation vector to degrees for readability
                    avg_rvec_degs = [round(math.degrees(x), 3) for x in avg_rvec]
                    avg_tvec_rounded = [round(x, 3) for x in avg_tvec]

                    cv2.putText(
                        expanded_image,
                        "Avg Marker ID: {}".format(marker_id),
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1
                    )
                    cv2.putText(
                        expanded_image,
                        "Pos: {}".format(avg_tvec_rounded),
                        (10, y_offset + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1
                    )
                    cv2.putText(
                        expanded_image,
                        "Rot: {}".format(avg_rvec_degs),
                        (10, y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1
                    )
                    y_offset += 50

                # The final image to display with overlays and text
                cv_image = expanded_image

            except KeyboardInterrupt:
                pass

        # Return the average results, this-frame detections, and the annotated image
        return self.current_average, cv_image