"""
NAO Robot Playing the Xylophone -- Main Client
Group C: Zhiyu Wang, Yijia Qian, Yuan Cao

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! READ BEFORE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
1.  EMERGENCY STOP -- BUMPER of the RIGHT FOOT / "q" key on the keyboard in the camera window
    If an abnormality occurs at any time, you can use any object to touch the bumper, and the robot will try to return
    to the initial position and relieve the stiffness of the whole body.
2.  Please always maintain a certain clearance with the robot while it is running.
3.  Make sure the robot posture is appropriate and the xylophone is placed in a suitable position in front of the robot,
    keeping a certain distance from its limbs and torso to avoid collision and interference.
4.  STOP as soon as possible if the robot reported motor overheating.
    After the robot overheats, the posture will be uncontrollable and there is a risk of felling down.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Requirements:
NAOqi Python SDK Version: 2.1
Libraries: numpy, opencv

This script is the main client program.
In this program, the NAO robot can perform the following tasks:
1.  Initialize various interface programs of NAOqi API.
2.  Make the robot enter the crouch posture and make all preparations, including: video stream subscription,
    audio stream subscription, voice command recognition stream subscription.
3.  Touching the middle button on the top of the robot's head can trigger the robot's voice command recognition
    function. Touching different parts of the body can also make it perform the following tasks.
4.  Detect ArUco marker [41] and calculate the grasp positions by transposing the marker posture to the target hand
    postures in the robot's torso frame.
5.  Detect ArUco marker [11, 21, 31] and mapping the key positions, then calculate the corresponding hand positions
    and wrist_yaw angles w.r.t each note.
6.  Play the xylophone by using the hand positions (w.r.t notes) and the musical sheets in a csv file.
7.  Listen to the notes (played by human with the xylophone) and write the notes into a csv file.

Quick Guide:
1.  Make sure the robot is in crouch position and start the program. The robot will start initializing.
2.  After hearing the robot's voice feedback "ready", you can instruct the robot to perform various tasks in the
    following two ways:
    a.  Touch the robot's head middle button to trigger the voice command recognition.
        1)  Say "grasp" to start grasping the mallets first.
        2)  Say "check" to calculate the keyposition.
        3)  Say "play" to start playing the xylophone.
        4)  Say "listen" to start listening to the notes.
        5)  Say "replay" to replay the listened melody.
        6)  Say "rest" to stop the program.
        7)  Give "yes" or "no" feedback to confirm the command.
    b.  Touch the robot's body parts to trigger the corresponding functions.
        1)  Touch the right hand back to start grasping the mallets.
        2)  Touch the head front to calculate the keyposition.
        3)  Touch the left hand back to start playing the xylophone.
        4)  Touch the head rear to start listening to the notes.

Client Developer: Zhiyu Wang
for the course "Humanoid Robotics System" as the final project
"""
from __future__ import print_function

import time
import cv2
import threading
import Queue
import csv
import numpy as np

from naoqi import ALProxy, ALModule, ALBroker

from aruco_marker import ArucoDetector
from nao_service import robot_init
from grasp_control import calculate_grasp_positions, execute_grasp_sequence
from play_xylophone import playMelodyFromCSV
from compute_keyposi import compute_keyposition
from pitch_detection import audio_feedback


# Parameters 
camerapara_dict = {0:np.array([]), 1:np.array([
            [286.6866057, 0.000000, 162.2270394],
            [0.000000, 285.2998696, 105.4745985], 
            [0.000000, 0.000000, 1.000000]
        ], dtype=np.float32), 2:np.array([
            [562.3129566280151, 0.000000, 324.21437038284097],
            [0.000000, 555.8965725951891, 217.4121628010703], 
            [0.000000, 0.000000, 1.000000]
        ], dtype=np.float32), 3:np.array([
            [1147.444868, 0.000000, 644.3633676],
            [0.000000, 1143.986342, 430.8996752], 
            [0.000000, 0.000000, 1.000000]
        ], dtype=np.float32)}
cameradist = np.array([-0.06665819242416217,0.09060075882427537,-0.00012550218643474006,
                       -0.0012131476680471336,-0.05834098541272104], dtype=np.float32)

ReactToTouch = None
memory = None


class ReactToTouch(ALModule):
    def __init__(self, name, stop_event, get_path_event, grasp_event, grasp_action, key_calculate_event, play_event,
                 notelisten_event, name_queue):
        ALModule.__init__(self, name)
        # Create the necessary proxies
        self.tts = ALProxy("ALTextToSpeech")
        self.asr = ALProxy("ALSpeechRecognition")
        self.anm = ALProxy("ALAutonomousMoves")
        
        # Store the events and queue
        self.stop_event             = stop_event
        self.get_path_event         = get_path_event
        self.grasp_event            = grasp_event
        self.grasp_action           = grasp_action
        self.key_calculate_event    = key_calculate_event
        self.play_event             = play_event
        self.notelisten_event       = notelisten_event
        
        self.name_queue = name_queue
       
        self.asr.setLanguage("English") 
        vocabulary = ["play", "replay", "listen", "grasp", "rest", "check",     # Commands
                      "joy", "star",                                            # Songs
                      "yes", "no"]
        self.asr.setVocabulary(vocabulary, False)
        self.recognized_command = []
        self.recognized_confirm = []
        self.recognized_song = []

        # Subscribe to TouchChanged event:
        global memory
        memory = ALProxy("ALMemory")
        memory.subscribeToEvent("TouchChanged",
                                "ReactToTouch",
                                "onTouched")
        

    def onTouched(self, strVarName, value):
        # Unsubscribe to the event when talking,
        # to avoid repetitions
        memory.unsubscribeToEvent("TouchChanged",
                                  "ReactToTouch")

        touched_bodies = []
        for p in value:
            if p[1]:
                touched_bodies.append(p[0])

        # Remove all the events
        if "LFoot/Bumper/Left" in touched_bodies:
            self.tts.stopAll()
            self.asr.unsubscribe("task")
            self.memory.removeData("WordRecognized")
        
        # right foot bumper touched, force shut down for emergency
        if "RFoot/Bumper/Right" in touched_bodies:
            self.stop_event.set()
        
        # right hand touched, start grasping
        if "RHand/Touch/Back" in touched_bodies:
            self.grasp_event.clear()
            self.get_path_event.set()
            self.grasp_action.set()
            
        # left hand touched, start playing
        if "LHand/Touch/Back" in touched_bodies:
            self.play_event.set()
            
        # head front touched, start calculating keyposition   
        if "Head/Touch/Front" in touched_bodies:
            self.key_calculate_event.set()
        
        # head middle touched, start speech recognition
        if "Head/Touch/Middle" in touched_bodies:
            self.speechrecognition()
            pass
        
        # head rear touched, start listening to the notes
        if "Head/Touch/Rear" in touched_bodies:
            self.notelisten_event.set()
            time.sleep(15)
            self.notelisten_event.clear()

        # Subscribe again to the event
        memory.subscribeToEvent("TouchChanged",
            "ReactToTouch",
            "onTouched")

    def check_answer(self):
        # Check the recognized command
        listened = []
        self.recognized_confirm = []
        confirmed = False

        while confirmed is not True:
            self.asr.subscribe("check_answer")
            time.sleep(2.5)
            listened = memory.getData("WordRecognized")
            if listened[1] > 0.3:
                self.recognized_confirm.append(str(listened[0]))
            self.asr.unsubscribe("check_answer")
            if self.recognized_confirm:
                confirmed = True
                if self.recognized_confirm[0] == "yes":
                    return True
                else:
                    return False
            else:
                self.tts.say("I didn't understand, please try again.")

            
    
    
    def speechrecognition(self):
        listened = []
        self.recognized_command = []
        self.recognized_song = []
        
        # !!! NEVER set the expressive listening to True !!! 
        self.anm.setExpressiveListeningEnabled(False)
        # Otherwise it will cause the robot to jump up

        # Subscribe to the event, start listening
        self.asr.subscribe("command")
        time.sleep(2.5)
        listened = memory.getData("WordRecognized")
        if listened[1]>0.3:
            self.recognized_command.append(str(listened[0]))
        self.asr.unsubscribe("command")

        # Check the recognized command
        if self.recognized_command:
            check_sentence = "Do you want me to " + self.recognized_command[0] + ", yes or no?"
            self.tts.say(check_sentence)
            check_result = self.check_answer()
        else:
            self.tts.say("I didn't understand.")
            check_result = False
            pass

        # if the recognized command is correct, then execute the command
        if check_result is True:
            # Ask for the song name and play the song
            if self.recognized_command[0] == "play":
                self.tts.say("Which song do you want me to play?")
                self.asr.subscribe("play")
                time.sleep(2.5)
                listened = memory.getData("WordRecognized")
                if listened[1] > 0.3:
                    self.recognized_song.append(str(listened[0]))
                self.asr.unsubscribe("play")
                
                song_name = ""
                with open('./src/support_service/content.csv', 'rb') as csvcontent:
                    reader = csv.reader(csvcontent)
                    for row in reader:
                        if row[0] == self.recognized_song[0]:
                            song_name = row[1]
                            break
                        
                self.tts.say("Do you want me to play "  + song_name + ", yes or no?")
                check_result_2 = self.check_answer()
                if check_result_2 is True:
                    self.tts.say("I will play " + song_name)
                    self.name_queue.put(self.recognized_song[0])
                    self.play_event.set()
            # Ask for the grasp action, if the mallets are not grasped, then grasp them
            elif self.recognized_command[0] == "grasp":
                if not self.grasp_event.is_set():
                    self.get_path_event.set()
                    self.grasp_action.set()
                else:
                    self.tts.say("I've already grasped both sticks, Do you want me to do it again?")
                    check_result_3 = self.check_answer()
                    if check_result_3 is True:
                        self.grasp_event.clear()
                        self.get_path_event.set()
                        self.grasp_action.set()
            # Execute the commands
            elif self.recognized_command[0] == "rest":
                self.stop_event.set()
            elif self.recognized_command[0] == "listen":
                self.notelisten_event.set()
                time.sleep(15)
                self.notelisten_event.clear()
            elif self.recognized_command[0] == "replay":
                self.name_queue.put("replay")
                self.play_event.set()
            elif self.recognized_command[0] == "check":
                self.key_calculate_event.set()
        else:
            self.tts.say("Sorry, try again please.")
        
        
def capture_video(robot_ip, robot_port, data_queue, camera_id, resolution, stop_event):
    """
    Continuously capture frames from NAO's camera, detect ArUco markers,
    and push results into data_queue.
    """
    video_proxy = ALProxy("ALVideoDevice", robot_ip, robot_port)
    global camerapara_dict
    global cameradist

    name_id = "python_video_live_stream"
    video_client = video_proxy.subscribeCamera(name_id, camera_id, resolution, 13, 30)   # 11-RGB 13-BGR, 30 Fps
    detector = ArucoDetector(camerapara_dict[resolution], cameradist)
    
    try:    
        while not stop_event.is_set():
            nao_image = video_proxy.getImageRemote(video_client)
            
            if nao_image is None:
                # Could be no frame available, just skip this iteration or print a warning
                print("No image received from the camera.")
                continue
            else: 
                width = nao_image[0]
                height = nao_image[1]
                array = nao_image[6]
            
            # Convert the image data to a numpy array
            image = np.frombuffer(bytearray(array), dtype=np.uint8).reshape((height, width, 3))
            results, processed_image = detector.detect_and_transform(image)
            if data_queue.full():
                data_queue.get()
                
            data_queue.put(results)
            
            # Display the image
            cv2.imshow("NAO Bottom Video Stream", processed_image)
            
            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    
    finally: 
        video_proxy.unsubscribe(video_client)
        cv2.destroyAllWindows()
        print("Video streaming stopped.")
    
        
def main(robot_ip="10.152.246.194", robot_port=9559):
    asrBroker = ALBroker("asrBroker",
       "0.0.0.0",           # listen to anyone
       0,                   # find a free port and use it
       robot_ip,            # parent broker IP
       robot_port)          # parent broker port
    tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
    
    # Thread control events
    stop_event = threading.Event()          # Control the full stop of the program
    get_path_event = threading.Event()      # get the path or not from the path_queue for calculating the grasp position
    grasp_event = threading.Event()         # Grasp trigger
    grasp_action = threading.Event()        # Start the grasp sequence
    key_calculate_event = threading.Event() # Calculate the keyposition
    play_event = threading.Event()          # If set, nao start to play the melody
    notelisten_event = threading.Event()    # If set, the listened notes will writen into a csv file
    
    # Data queues
    data_queue = Queue.Queue()              # ArUco Marker position and orientation data Stream
    path_queue = Queue.Queue()              # Grasp position data Stream   
    note_queue = Queue.Queue()              # Note data Stream
    name_queue = Queue.Queue()              # Name of the sheet to play
    
    # Video parameters
    camera_id = 1          # 0=Top, 1=Bottom
    resolution = 2         # 0=160x120, 1=320x240, 2=640x480, 3=1280x960

    # Threads 1: Robot Initialization
    robot_init_thread = threading.Thread(target=robot_init, args=(robot_ip, robot_port,
                                                                  stop_event))
    # Threads 2: Video Stream and ArUco Marker Posture Detection
    video_thread = threading.Thread(target=capture_video, args=(robot_ip, robot_port, data_queue, camera_id, resolution, 
                                                                stop_event))
    # Threads 3: Audio Stream and Pitch Detection
    audio_thread = threading.Thread(target=audio_feedback, args=(robot_ip, robot_port,
                                                                 stop_event, notelisten_event, note_queue))
    # Threads 4: Grasp Position Calculation by using the marker[41] posture
    grasp_calculation_thread = threading.Thread(target=calculate_grasp_positions, args=(data_queue, path_queue,
                                                                                        stop_event, get_path_event,
                                                                                        grasp_action))
    # Threads 5: Grasp Action by using the grasp position
    hand_control_thread = threading.Thread(target=execute_grasp_sequence, args=(robot_ip, robot_port, path_queue,
                                                                                stop_event, get_path_event,
                                                                                grasp_event, grasp_action))
    # Threads 6: Key Positions Calculation by using the marker[11,21,31] positions
    compute_keyposi_thread = threading.Thread(target=compute_keyposition, args=(data_queue,
                                                                                stop_event, key_calculate_event))
    # Threads 7: Xylophone Playing by using the key positions and the melody csv file
    play_thread = threading.Thread(target=playMelodyFromCSV, args=(robot_ip, robot_port, name_queue,
                                                                   stop_event, play_event))
    
    
    # Robot initialization
    robot_init_thread.start()
    time.sleep(5)               # Wait for the robot to initialize
    video_thread.start()
    audio_thread.start()
    time.sleep(2)               # Wait for the video and audio thread to initialize
    grasp_calculation_thread.start()
    hand_control_thread.start()
    compute_keyposi_thread.start()
    play_thread.start()
    tts.say("I am ready.")

    # Initialize the User Interface
    global ReactToTouch
    ReactToTouch = ReactToTouch("ReactToTouch", stop_event, get_path_event, grasp_event, 
                                                grasp_action, key_calculate_event, play_event, 
                                                notelisten_event, name_queue)
    
    try:
        while True:
            time.sleep(1)
            if stop_event.is_set():
                break
    except KeyboardInterrupt:
        print("Ctrl+C detected. Shutting down...")
        stop_event.set()

    # Join all threads
    robot_init_thread.join()
    video_thread.join()
    audio_thread.join()
    grasp_calculation_thread.join()
    hand_control_thread.join()
    compute_keyposi_thread.join()
    play_thread.join()

    print("Program stopped.")


if __name__ == "__main__":
    robot_ip    = "10.152.246.194"
    robot_port  = 9559
    
    print("Connecting to NAO robot at IP {} on port {}.".format(robot_ip, robot_port))
    main(robot_ip, robot_port)
    