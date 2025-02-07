"""
Pitch Detection with Real-Time Visualization with Custom Audio Source
Group C: Zhiyu Wang, Yijia Qian, Yuan Cao

Requirements:
Python Version: 3.11 or higher
Libraries: numpy, matplotlib, sounddevice, librosa (recommended)
Librosa is recommended for optimized pitch detection, but FFT-based detection is also available.

This is a demonstration of a real-time pitch detection system with optimized performance.
This script does not require NAOqi SDK and can be run on any Python environment or devices that have a microphone.
Though the algorithm has been used in the NAO robot project, here is to demonstrate the potentials of the pitch
detection with:
1. higher sampling rate,
2. lower latency,
3. optimized pitch detection,
4. real-time visualization.

Developer: Zhiyu Wang
for the course "Humanoid Robotics System" as the final project: Pitch Detection
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import time
from collections import deque

# Optional librosa installation: pip install librosa (not supported for Python 2.7)
use_librosa = True
try:
    import librosa
except ImportError:
    use_librosa = False


class AudioSource:
    """Base class for generic audio sources"""

    def get_block_data(self):
        raise NotImplementedError

    def stop(self):
        pass


class SoundDeviceAudioSource(AudioSource):
    """Optimized sounddevice audio source"""

    def __init__(self, fs=44100, channels=1, block_duration=20):  # Reduced block duration for lower latency
        self.fs = fs
        self.channels = channels
        self.block_duration = block_duration
        self.block_size = int(fs * block_duration / 1000)
        self.latest_data = np.zeros(self.block_size, dtype='float32')

        import sounddevice as sd
        self.stream = sd.InputStream(
            callback=self._audio_callback,
            channels=self.channels,
            samplerate=self.fs,
            blocksize=self.block_size,
            latency='low'  # Enable low-latency mode
        )
        self.stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        self.latest_data = indata[:, 0].copy()  # Directly store latest data

    def get_block_data(self):
        return self.latest_data

    def stop(self):
        self.stream.stop()
        self.stream.close()


class RealTimeNoteProcessor:
    def __init__(self, source, fs=44100, block_duration=20, display_duration=2.0,
                 pitch_threshold=100.0, amplitude_threshold=0.01):
        self.source = source
        self.fs = fs
        self.block_duration = block_duration  # ms
        self.display_duration = display_duration  # s
        self.block_size = int(fs * block_duration / 1000)
        self.buffer_size = int(fs * display_duration)

        # Audio buffer using deque for better performance
        self.waveform_buffer = deque(maxlen=self.buffer_size)
        self.waveform_buffer.extend(np.zeros(self.buffer_size))

        # Pitch detection parameters
        self.pitch_threshold = pitch_threshold
        self.amplitude_threshold = amplitude_threshold
        self.use_librosa = use_librosa

        # Note tracking state
        self.current_note_pitch = None
        self.current_note_start_time = None
        self.detected_notes_raw = []

        # Initialize visualization
        self.init_plots()

        # Use timer instead of FuncAnimation for better responsiveness
        self.timer = self.fig.canvas.new_timer(interval=self.block_duration)
        self.timer.add_callback(self.update)
        self.timer.start()

    def init_plots(self):
        """Initialize optimized visualization"""
        self.fig, (self.ax_time, self.ax_pitch) = plt.subplots(2, 1, figsize=(10, 6))

        # Time domain settings
        self.time_axis = np.linspace(0, self.display_duration, self.buffer_size)
        self.line_time, = self.ax_time.plot(self.time_axis, np.zeros(self.buffer_size))
        self.ax_time.set_ylim(-1, 1)
        self.ax_time.set_title('Real-Time Audio Waveform')

        # Pitch display settings
        self.pitch_line = self.ax_pitch.axhline(0, color='r', linestyle='--')
        self.note_text = self.ax_pitch.text(0.8, 0.9, '', transform=self.ax_pitch.transAxes)
        self.ax_pitch.set_ylim(0, 2000)
        self.ax_pitch.set_title('Instant Pitch Detection')

        plt.tight_layout()

    def freq_to_note_name(self, f):
        if f < self.pitch_threshold:
            return "N/A"
        try:
            note_number = 69 + 12 * math.log2(f / 440.0)
            return f"{['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][int(round(note_number) % 12)]}{int(round(note_number) / 12) - 1}"
        except:
            return "N/A"

    def estimate_pitch(self, audio_block):
        """Optimized pitch estimation method"""
        # Quick amplitude check
        if np.mean(np.abs(audio_block)) < self.amplitude_threshold:
            return 0.0

        # Method selection based on availability
        if self.use_librosa:
            return self.librosa_pitch(audio_block)
        else:
            return self.fft_pitch(audio_block)

    def fft_pitch(self, audio_block):
        """Optimized FFT detection"""
        N = len(audio_block)
        window = np.hamming(N)
        fft = np.fft.rfft(audio_block * window)
        freqs = np.fft.rfftfreq(N, 1 / self.fs)
        peak_idx = np.argmax(np.abs(fft)[1:]) + 1  # Skip DC component
        return freqs[peak_idx]

    def librosa_pitch(self, audio_block):
        """Optimized librosa detection"""
        audio_block = audio_block.astype(np.float32)
        f0 = librosa.yin(
            audio_block,
            fmin=80,  # Raise minimum detection frequency
            fmax=2000,
            sr=self.fs,
            frame_length=1024,  # Reduce frame length
            hop_length=256  # Increase hop length
        )
        valid_f0 = f0[(f0 > 80) & (f0 < 2000)]
        return np.median(valid_f0) if len(valid_f0) > 0 else 0.0

    def update(self):
        """Optimized update loop"""
        try:
            # Acquire latest audio block
            new_data = self.source.get_block_data()

            # Update waveform buffer
            self.waveform_buffer.extend(new_data)

            # Real-time pitch detection (using only latest block)
            current_pitch = self.estimate_pitch(new_data)

            # Update note state
            self.update_note_detection(current_pitch)

            # Refresh visualization
            self.update_plots(np.array(self.waveform_buffer), current_pitch)

            # Force immediate redraw
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        except Exception as e:
            print(f"Update error: {e}")

    def update_note_detection(self, pitch):
        """Optimized note detection logic"""
        current_time = time.time()

        # Valid pitch determination
        valid = (pitch > self.pitch_threshold and
                 np.mean(np.abs(self.waveform_buffer)) > self.amplitude_threshold)

        if valid:
            if self.current_note_pitch is None:
                # Start new note
                self.current_note_pitch = pitch
                self.current_note_start_time = current_time
            else:
                # Pitch change detection
                if abs(pitch - self.current_note_pitch) > 50:
                    self.finalize_note(current_time)
                    self.current_note_pitch = pitch
                    self.current_note_start_time = current_time
        else:
            if self.current_note_pitch is not None:
                self.finalize_note(current_time)

    def finalize_note(self, end_time):
        """Finalize note recording"""
        duration = end_time - self.current_note_start_time
        quantized = self.quantize_duration(duration)
        if quantized:
            self.detected_notes_raw.append((
                self.current_note_pitch,
                quantized
            ))
            self.export_to_csv()
        self.current_note_pitch = None

    def quantize_duration(self, duration):
        """Optimized quantization logic"""
        base = 0.125  # Minimum quantization unit (eighth note)
        quantized = round(duration / base) * base
        return quantized if quantized >= base else None

    def update_plots(self, waveform, pitch):
        """Optimized plot updates"""
        # Time domain waveform update
        self.line_time.set_ydata(waveform)

        # Pitch display update
        self.pitch_line.set_ydata([pitch, pitch])
        self.note_text.set_text(
            f"{self.freq_to_note_name(pitch)}\n{pitch:.1f} Hz"
            if pitch > self.pitch_threshold else "No Note"
        )

    def export_to_csv(self, filename="replay.csv"):
        """Optimized export logic"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Note", "Duration"])
            for pitch, duration in self.detected_notes_raw[-10:]:  # Keep only last 10 notes
                writer.writerow([self.freq_to_note_name(pitch), duration])

    def run(self):
        plt.show()


def main():
    # Use low-latency configuration
    source = SoundDeviceAudioSource(fs=44100, block_duration=20)
    processor = RealTimeNoteProcessor(
        source=source,
        fs=44100,
        block_duration=20,
        pitch_threshold=100,
        amplitude_threshold=0.02
    )
    processor.run()
    source.stop()


if __name__ == "__main__":
    main()