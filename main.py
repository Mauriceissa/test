import datetime
import json
import logging
import logging.handlers
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyaudio
import resampy
import requests
import wave
from scipy.io import wavfile as wav
from tflite_runtime.interpreter import Interpreter
import librosa

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "model.tflite"
CAPTURE_RATE = 44100  # Microphone records at 44100 Hz
MODEL_RATE = 16000    # TFLite model expects 16000 Hz
CHUNK_SIZE = 1024
AUDIO_LENGTH = 1.0    # 1 second of audio
DEVICE_INDEX = 11     # USB Audio Device (hw:2,0)

# Global variables
audio_queue = queue.Queue()
stop_event = threading.Event()

def load_model(model_path):
    """Load the TensorFlow Lite model."""
    try:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        logger.info(f"Model loaded successfully from {model_path}")
        return interpreter
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def preprocess_audio(audio_data, sample_rate):
    """Preprocess audio data for model inference."""
    try:
        audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != MODEL_RATE:
            audio_data = resampy.resample(audio_data, sample_rate, MODEL_RATE)
        audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
        return audio_data
    except Exception as e:
        logger.error(f"Error in preprocessing audio: {e}")
        raise

def predict(interpreter, audio_data):
    """Run inference on the preprocessed audio data."""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], audio_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise

def audio_callback(in_data, frame_count, time_info, status):
    """Callback for PyAudio to handle audio input."""
    if status:
        logger.warning(f"Audio callback status: {status}")
    print(f"Captured audio chunk of size {len(in_data)} bytes")
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def audio_thread():
    """Run audio capture in a separate thread."""
    try:
        p = pyaudio.PyAudio()
        print("PyAudio initialized")
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=CAPTURE_RATE,  # Match microphone rate
                        input=True,
                        input_device_index=DEVICE_INDEX,  # USB mic
                        frames_per_buffer=CHUNK_SIZE,
                        stream_callback=audio_callback)
        print(f"Audio stream opened with device index {DEVICE_INDEX}")
        stream.start_stream()
        print("Audio stream started")
        while not stop_event.is_set():
            time.sleep(0.1)
        stream.stop_stream()
        print("Audio stream stopped")
        stream.close()
        p.terminate()
        print("PyAudio terminated")
    except Exception as e:
        logger.error(f"Error in audio thread: {e}")
        stop_event.set()
        raise

def main():
    """Main function to run the audio recognition pipeline."""
    # Disable PulseAudio to avoid interference
    try:
        subprocess.run(["pulseaudio", "--kill"], check=True)
        print("PulseAudio stopped")
    except Exception as e:
        logger.warning(f"Could not stop PulseAudio: {e}")

    try:
        interpreter = load_model(MODEL_PATH)
        thread = threading.Thread(target=audio_thread)
        thread.start()

        while not stop_event.is_set():
            try:
                audio_data = audio_queue.get(timeout=1.0)
                processed_audio = preprocess_audio(audio_data, CAPTURE_RATE)
                if len(processed_audio) > MODEL_RATE * AUDIO_LENGTH:
                    processed_audio = processed_audio[:int(MODEL_RATE * AUDIO_LENGTH)]
                else:
                    processed_audio = np.pad(processed_audio, (0, int(MODEL_RATE * AUDIO_LENGTH) - len(processed_audio)))
                processed_audio = processed_audio.reshape(1, -1, 1)
                prediction = predict(interpreter, processed_audio)
                logger.info(f"Prediction: {prediction}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
    except KeyboardInterrupt:
        logger.info("Stopping...")
        stop_event.set()
        thread.join()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        stop_event.set()
        thread.join()
        sys.exit(1)

if __name__ == "__main__":
    main()
