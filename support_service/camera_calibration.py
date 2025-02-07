#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
NAO Robot Top/Bottom Camera Calibration
Group C: Zhiyu Wang, Yijia Qian, Yuan Cao

Requirements:
Naoqi Python SDK Version: 2.1
Libraries: numpy, cv2, csv

This script connects to the NAO robot's bottom camera, captures multiple images
of a checkerboard pattern, detects the checkerboard corners, and performs camera
calibration to compute the intrinsic parameters and distortion coefficients.
This script runs standalone.

Outputs:
- Calibration parameters printed in the command line.
- Calibration parameters saved to 'file_name.csv'.

Developer: Zhiyu Wang
for the course "Humanoid Robotics System" as the final project
"""

import cv2
import numpy as np
import time
import sys
import csv
from naoqi import ALProxy

def main():
    # === Configuration Parameters ===
    robot_ip = "10.152.246.194"  # Replace with your NAO's IP address
    robot_port = 9559            # Default NAOqi port

    video_module_name = "BottomCameraCalibration"
    # Video parameters
    camera_id = 1          # 0=Top, 1=Bottom
    resolution = 3         # 0=160x120, 1=320x240, 2=640x480, 3=1280x960
    color_space = 11       # 11=RGB, 13=BGR
    fps = 15               # Frames per second

    checkerboard_size = (8, 6)     # Number of internal corners (columns, rows)
    square_size = 50               # Size of a square in mm
    num_images = 20                # Number of calibration images to capture

    calibration_csv = 'bottom_960p_cali3.csv'  # Output CSV file

    # === Initialize ALVideoDevice Proxy ===
    try:
        video_proxy = ALProxy("ALVideoDevice", robot_ip, robot_port)
        print("[INFO] Connected to ALVideoDevice on {}:{}".format(robot_ip, robot_port))
    except Exception as e:
        print("[ERROR] Could not create ALVideoDevice proxy.")
        print("[ERROR] Exception:", e)
        sys.exit(1)

    # === Subscribe to the Bottom Camera ===
    try:
        subscription_id = video_proxy.subscribeCamera(video_module_name, camera_id, resolution, color_space, fps)
        print("[INFO] Subscribed to camera '{}' with subscription ID: {}".format(camera_id, subscription_id))
    except Exception as e:
        print("[ERROR] Could not subscribe to camera.")
        print("[ERROR] Exception:", e)
        sys.exit(1)

    # === Prepare Object Points ===
    objp = np.zeros((checkerboard_size[1] * checkerboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0],
                           0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale by square size to get real-world coordinates

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Create a window to display images
    cv2.namedWindow('Bottom Camera Calibration', cv2.WINDOW_AUTOSIZE)

    print("\n=== Camera Calibration Process ===")
    print("Please present the checkerboard to the bottom camera and ensure it is fully visible.")
    print("Press 'q' to quit early.\n")

    captured_images = 0
    while captured_images < num_images:
        # === Capture Image ===
        try:
            nao_image = video_proxy.getImageRemote(subscription_id)
            if nao_image is None:
                print("[WARNING] Failed to capture image from camera.")
                continue

            # Extract image dimensions
            width = nao_image[0]
            height = nao_image[1]
            array = nao_image[6]  # Pixel data

            # Convert to NumPy array
            img = np.frombuffer(bytearray(array), dtype=np.uint8).reshape(height, width, 3)

            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("[ERROR] Error capturing image:", e)
            continue

        # === Convert to Grayscale ===
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # === Find Checkerboard Corners ===
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        # If found, add object points and image points
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            captured_images += 1
            print("[INFO] Captured image {}/{}".format(captured_images, num_images))
        else:
            print("[INFO] Checkerboard not detected in the current frame.")

        # Display the image
        cv2.imshow('Bottom Camera Calibration', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Early termination requested by user.")
            break

        # Optional: Add a delay to allow user to reposition the checkerboard
        time.sleep(1)

    # === Unsubscribe from the Camera ===
    video_proxy.unsubscribe(subscription_id)
    cv2.destroyAllWindows()
    print("\n[INFO] Image capture completed.")

    if len(objpoints) < 10:
        print("[WARNING] Only {} valid images captured. More images may improve calibration accuracy.".format(len(objpoints)))

    if len(objpoints) == 0:
        print("[ERROR] No valid images were captured for calibration.")
        sys.exit(1)

    # === Perform Camera Calibration ===
    print("\n[INFO] Performing camera calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )

    if ret:
        print("[SUCCESS] Calibration was successful!")
        print("\n=== Camera Matrix ===\n", camera_matrix)
        print("\n=== Distortion Coefficients ===\n", dist_coeffs.ravel())

        # === Save Calibration Parameters to CSV ===
        try:
            with open(calibration_csv, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)

                # Write Camera Matrix
                csvwriter.writerow(['Camera Matrix'])
                for row in camera_matrix:
                    csvwriter.writerow(row)
                csvwriter.writerow([])  # Empty line

                # Write Distortion Coefficients
                csvwriter.writerow(['Distortion Coefficients'])
                csvwriter.writerow(dist_coeffs.ravel())

            print("\n[INFO] Calibration parameters saved to '{}'.".format(calibration_csv))
        except Exception as e:
            print("[ERROR] Could not save calibration parameters to CSV.")
            print("[ERROR] Exception:", e)

    else:
        print("[FAILURE] Calibration failed.")

if __name__ == '__main__':
    main()
