"""
NAO Robot Services Module
Group C: Zhiyu Wang, Yijia Qian, Yuan Cao

Requirements:
NAOqi Python SDK Version: 2.1
No other libraries are required in this script.

This script provides a service to initialize the robot and set it to Crouch posture.
While stop_event is set, the robot will stop the service and shutdown.

Developer: Zhiyu Wang
for the course "Humanoid Robotics System" as the final project
"""
from naoqi import ALProxy, ALModule, ALBroker
import time
import sys

def robot_init(robot_ip, robot_port, stop_event):
    """
    Initialize the robot and keep the thread alive until stop_event is set.
    """
    motionProxy = ALProxy("ALMotion", robot_ip, robot_port)
    postureProxy = ALProxy("ALRobotPosture", robot_ip, robot_port)
    tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
    
    # Wake up and posture
    motionProxy.wakeUp()
    postureProxy.goToPosture("Crouch", 0.5)

    try:
        while not stop_event.is_set():
            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        tts.say("I am going to rest now.")
        motionProxy.rest()
        sys.exit(0)
