#!/usr/bin/env python
# -*- encoding: UTF-8 -*-

"""
Pitch detection and note recognition using the ALAudioDevice module
Group C: Zhiyu Wang, Yijia Qian, Yuan Cao

Requirements:
NAOqi API Version: 2.8 (ALSoundDevice is not working in version 2.1)
Libraries: numpy, scipy

This Script mainly uses the ALAudioDevice module to process audio data from the robot's microphones.
The raw data is a 16-bit little-endian audio stream.
Then estimates the fundamental frequency of the audio signal using FFT and detects notes based on the pitch.
The detected notes are then exported to a CSV file for further processing by demand through an event flag set by the
main client.
This script can be used as a standalone module.
Audio device can be substituted with other audio sources for testing purposes, such as microphone input from a computer.

Notes: Since librosa is not supported in python 2.7, we use numpy and scipy to estimate the pitch.
With librosa, the pitch estimation can be more accurate and efficient, also have a good robustness.

Developer: Zhiyu Wang
for the course "Humanoid Robotics System" as the final project: Task 5 and 6
"""

import qi
import sys
import time
import numpy as np
import math
import csv


class SoundProcessingModule(object):
    def __init__(self, app, notelisten_event, stop_event):
        super(SoundProcessingModule, self).__init__()
        app.start()
        session = app.session
        self.audio_service = session.service("ALAudioDevice")
        self.isProcessingDone = False
        self.micFront = []
        self.module_name = "SoundProcessingModule"
        self.fs = 16000                                     # Sampling frequency
        self.channels = 1                                   # Mono
        self.block_duration = 85.3125                       # in milliseconds
        self.block_size = 1365                              # Number of samples per block (seems defined by the robot)
        self.display_duration = 2.0                         # Duration to display in seconds
        self.buffer_size = int(self.fs * self.display_duration)
        self.pitch_threshold = 50.0                         # Minimum pitch to consider
        self.amplitude_threshold = 0.01                     # Minimum amplitude to consider

        self.detected_notes_raw = []                        # List to store detected notes
        self.num_blocks = int(self.display_duration * 1000 / self.block_duration)
        self.pitch_buffer = np.zeros(self.buffer_size)
        self.pitch_time = np.linspace(0, self.display_duration, self.num_blocks)
        self.pitch_index = 0

        self.waveform_buffer = np.zeros(self.buffer_size)
        self.time_axis = np.arange(0, self.buffer_size) / float(self.fs)
        self.notelisten_event = notelisten_event            # Event to control listening and recording notes in csv
        self.stop_event = stop_event                        # Event to stop the audio feedback

        # Variables to track the current note
        self.current_note_pitch = None
        self.current_note_start_time = None

    def startProcessing(self):
        """
        Continuously process audio data, detect pitches, and handle CSV recording based on the event state.
        When the event is set, start recording and overwrite the existing CSV file.
        When the event is cleared, stop recording but retain the CSV file.
        """
        self.audio_service.setClientPreferences(self.module_name, self.fs, self.channels, 0)
        self.audio_service.subscribe(self.module_name)

        # Track the previous state of the event to detect transitions
        last_event_state = False

        try:
            while not self.isProcessingDone and not self.stop_event.is_set():
                current_event_state = self.notelisten_event.is_set()

                # Detect transition from cleared to set
                if current_event_state and not last_event_state:
                    # Clear previous notes
                    self.detected_notes_raw = []
                    # Overwrite the CSV file by opening in write mode
                    # This effectively clears previous content without deleting the file
                    self.export_to_csv("replay.csv")
                    print("[INFO] Start new recording session, cleared old notes and overwrote CSV.")

                last_event_state = current_event_state

                # Estimate the fundamental frequency from the waveform buffer
                fundamental_freq = self.estimate_pitch_fft(self.waveform_buffer)

                # Update and detect notes based on the fundamental frequency
                self.update_note_detection(fundamental_freq)

                # If event is set and a valid pitch is detected, yield the note name
                if fundamental_freq > self.pitch_threshold:
                    note_name = self.freq_to_note_name(fundamental_freq)
                    yield note_name

                time.sleep(0.1)  # Sleep to prevent high CPU usage

        except KeyboardInterrupt:
            self.audio_service.unsubscribe(self.module_name)

    def processRemote(self, nbOfChannels, nbOfSamplesByChannel, timeStamp, inputBuffer):
        """
        Process incoming audio data from the remote source.
        """
        self.micFront = self.convertStr2SignedInt(inputBuffer)
        # Roll the waveform buffer and append new audio data
        self.waveform_buffer = np.roll(self.waveform_buffer, -self.block_size)
        self.waveform_buffer[-self.block_size:] = self.micFront

    def convertStr2SignedInt(self, data):
        """
        Convert a string containing 16-bit little-endian audio samples to a numerical vector ranging from -1 to 1.
        (int16) ==> (float32)
        """
        signedData = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        signedData /= 32768.0
        return signedData

    def freq_to_note_name(self, f):
        """
        Convert a frequency to its corresponding musical note name.
        """
        if f <= 0:
            return "N/A"
        # MIDI note number calculation (A4=440 Hz -> MIDI note 69)
        note_number = 69 + 12 * math.log(f / 440.0, 2)
        note_number_rounded = int(round(note_number))
        note_names = ["C", "C#", "D", "D#", "E", "F",
                      "F#", "G", "G#", "A", "A#", "B"]
        note_name = note_names[note_number_rounded % 12]
        octave = (note_number_rounded // 12) - 1
        return "%s%d" % (note_name, octave)

    def filter_instruments(self, pitch):
        if pitch < self.pitch_threshold:
            return False
        if pitch < 780 or pitch > 3150:
            return False
        return True

    def quantize_duration(self, duration):
        """
        Quantize the duration of a note to the nearest standard value.
        """
        possible_values = [1.0, 0.5, 0.25, 0.125, 0.0625]
        if duration < 0.0625:
            return None
        chosen_value = min(possible_values, key=lambda x: abs(x - duration))
        return chosen_value

    def normalize_durations(self):
        """
        Normalize the durations of detected notes based on the maximum duration.
        Considering the minimum duration can the NAO robot play, we set the minimum duration to 0.5s.
        As a result, the normalized duration will be d/min_d * 0.5.
        """
        if len(self.detected_notes_raw) == 0:
            return []
        durations = [n[1] for n in self.detected_notes_raw]  # Quantized durations
        min_d = min(durations)
        normalized = [(d / min_d) * 0.5 for d in durations]
        return normalized

    def export_to_csv(self, filename="notes.csv"):
        """
        Export the detected notes and their normalized durations to a CSV file.
        Overwrites the existing file when opened in write mode.
        """
        normalized = self.normalize_durations()
        names = []
        for pitch, _ in self.detected_notes_raw:
            names.append(self.freq_to_note_name(pitch))

        # Open the CSV file in write-binary mode for Python 2
        with open(filename, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["note", "lasting_time"])
            for n, d in zip(names, normalized):
                writer.writerow([n, d])

    def update_note_detection(self, fundamental_freq):
        """
        Update the note detection logic based on the current fundamental frequency.
        Only records notes when the event is set.
        """
        if not self.notelisten_event.is_set():
            # If the event is cleared, do not record notes
            return

        current_time = time.time()
        amplitude = np.mean(np.abs(self.waveform_buffer))
        valid = self.filter_instruments(fundamental_freq) and amplitude > self.amplitude_threshold

        if not valid:
            # If the current pitch is invalid, end any ongoing note
            if self.current_note_pitch is not None:
                note_duration = current_time - self.current_note_start_time
                quantized = self.quantize_duration(note_duration)
                if quantized is not None:
                    self.detected_notes_raw.append((self.current_note_pitch, quantized))
                    self.export_to_csv("notes.csv")  # Overwrite CSV with updated notes
                self.current_note_pitch = None
                self.current_note_start_time = None
            return

        if self.current_note_pitch is None:
            # Start a new note
            self.current_note_pitch = fundamental_freq
            self.current_note_start_time = current_time
        else:
            # If the pitch has changed significantly, end the previous note and start a new one
            if abs(self.current_note_pitch - fundamental_freq) > 50:
                note_duration = current_time - self.current_note_start_time
                quantized = self.quantize_duration(note_duration)
                if quantized is not None:
                    self.detected_notes_raw.append((self.current_note_pitch, quantized))
                    self.export_to_csv("notes.csv")  # Overwrite CSV with updated notes

                # Start the new note
                self.current_note_pitch = fundamental_freq
                self.current_note_start_time = current_time

    def estimate_pitch_fft(self, audio_buffer):
        """
        Estimate the fundamental frequency using FFT.
        """
        fft_data = np.fft.rfft(audio_buffer)
        magnitude = np.abs(fft_data)
        freq_axis = np.fft.rfftfreq(self.buffer_size, 1.0 / self.fs)
        max_idx = np.argmax(magnitude[1:]) + 1  # Ignore the zero frequency
        fundamental_freq = freq_axis[max_idx]
        return fundamental_freq

    def update(self, frame=None):
        """
        re-estimates the pitch and updates note detection.
        """
        fundamental_freq = self.estimate_pitch_fft(self.waveform_buffer)
        # Update the pitch buffer
        self.pitch_buffer = np.roll(self.pitch_buffer, -1)
        self.pitch_buffer[-1] = fundamental_freq if fundamental_freq > self.pitch_threshold else 0.0
        # Continue note detection
        self.update_note_detection(fundamental_freq)


def audio_feedback(robot_ip, robot_port, stop_event, notelisten_event, note_queue):
    """
    Function to handle audio feedback by instantiating the SoundProcessingModule and processing notes.
    """
    try:
        connection_url = "tcp://" + robot_ip + ":" + str(robot_port)
        app = qi.Application(["SoundProcessingModule", "--qi-url=" + connection_url])
    except RuntimeError or stop_event.is_set():
        print("[ERROR] Could not connect to the robot at {}:{}".format(robot_ip, robot_port))
        sys.exit(1)

    MySoundProcessingModule = SoundProcessingModule(app, notelisten_event, stop_event)
    app.session.registerService("SoundProcessingModule", MySoundProcessingModule)

    while not stop_event.is_set():
        # Iterate over detected notes and put them into the queue
        for note_name in MySoundProcessingModule.startProcessing():
            note_queue.put(note_name)
            # sys.stdout.write("\rNote: {}".format(note_name)) 
            # sys.stdout.flush()
            # Optionally perform additional updates or processing
            MySoundProcessingModule.update()


if __name__ == "__main__":
    robot_ip = "10.152.246.194"
    robot_port = 9559
    stop_event = False  
    notelisten_event = True  
    note_queue = [] 

    audio_feedback(robot_ip, robot_port, stop_event, notelisten_event, note_queue)
