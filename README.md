# Playing the Xylophone with NAO
Humanoid Robotic System -- Final Project
Group C: Zhiyu Wang, Yijia Qian, Yuan Cao

## Development Environment

### Method 1: Using Devcontainer or Docker Image of Python 2.7 with NAOqi SDK
1. Ensure Python 2.7 is installed.
2. Manually update `pip` using the following command:
   ```bash
   python2 get-pip.py
   ```

### Method 2: Manually Setting Up Python 2.7 Environment
1. Install Python 2.7 and NAOqi SDK version 2.1 by downloading from the following links based on your operating system:
   - **Linux:** [NAOqi SDK for Linux](https://corporate-internal-prod.aldebaran.com/en/support/nao-6/downloads-softwares/former-versions?os=49&category=76)
   - **Windows:** [NAOqi SDK for Windows](https://corporate-internal-prod.aldebaran.com/en/support/nao-6/downloads-softwares/former-versions?os=45&category=76)
   - **macOS:** [NAOqi SDK for macOS](https://corporate-internal-prod.aldebaran.com/en/support/nao-6/downloads-softwares/former-versions?os=47&category=76)

## Required Libraries
Ensure the following libraries are installed:

- `numpy==1.16.6`
- `opencv-python==4.2.0.32`
- `opencv-contrib-python==3.4.9.31`
- `pandas==0.24.2`
- `scipy==1.2.3`

---

## Important Safety Instructions

**READ BEFORE OPERATING THE ROBOT**

1. **Emergency Stop:** Press the bumper on the right foot or press the `q` key in the camera window.
   - If an issue arises, touch the bumper to reset the robot to its initial position and relieve full-body stiffness.

2. **Maintain a safe distance** from the robot while it is in operation.

3. **Ensure proper robot posture** and place the xylophone at an appropriate distance to avoid collisions.

4. **Monitor for motor overheating:**
   - If the robot reports overheating, stop the operation immediately to prevent it from falling due to uncontrollable posture.

---

## Quick Start Guide

### Steps to Start the Robot
1. Place the robot in a crouch position and start the `./src/client.py`. The robot will initialize.
    ```bash
   python ./src/client.py
   ```
2. Wait for the voice feedback "ready" and choose from the following interaction methods:

   **a. Using voice commands:**
   - Touch the middle button on the robot's head to activate voice recognition.
     1. Say **"grasp"** to grasp the mallets.
     2. Say **"check"** to calculate key positions.
     3. Say **"play"** to start playing the xylophone.
     4. Say **"listen"** to start listening to the notes.
     5. Say **"replay"** to replay the listened melody.
     6. Say **"rest"** to stop the program.
     7. Say **"yes"** or **"no"** to confirm commands.

   **b. Using physical touch commands:**
   - Touch different parts of the robot to trigger specific actions:
     1. Touch the right hand back to start grasping the mallets.
     2. Touch the head front to calculate key positions.
     3. Touch the left hand back to start playing the xylophone.
     4. Touch the head rear to start listening to the notes.

---

## Known Issues

1. **Accuracy Issues:** Due to potential motor or sensor overheating, joint positioning may be inaccurate, leading to slight deviations during performance. Manual adjustments of the robot's arm position may be required.
2. **Pitch Detection:** The pitch detection algorithm may not be 100% accurate, leading to incorrect note detection. The Real-time Pitch Correction feature is disabled to prevent incorrect note detection and the dead loop issue.

---