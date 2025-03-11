#!/usr/bin/env python3
"""
Hybrid Sound Detection System for Jetson Nano
- Records at 44100 Hz via USB microphone, resamples to 16000 Hz with librosa
- Combines robust mic selection, virtual mic support, and terminal feedback
- No GPS integration (reserved for version 2)
- Continuously records, classifies audio, and sends events to an API
"""

import datetime
import json
import logging
import logging.handlers
import os
import queue
import signal
import sys
import tensorflow as tf
import threading
import time
from pathlib import Path

import numpy as np
import pyaudio
import wave
import requests
from tflite_runtime.interpreter import Interpreter
import librosa

import config
import event_creation_IOT

# Configuration
DEVICE = config.DEVICE_SERIAL_NUMBER
RECORD_SECONDS = config.audioWindow  # Configurable from config
RATE = 44100  # Microphone records at 44100 Hz
MODEL_RATE = 16000  # Model expects 16000 Hz
CHANNELS = config.CHANNELS  # e.g., 1 for mono
CHUNK = config.CHUNK  # e.g., 1024
FORMAT = pyaudio.paInt16
FOLDER = config.cacheFolder
BASE_API_URL = config.BASE_API_URL
PREFERRED_DEVICE_INDEX = 11  # For USB Audio Device (card 2, device 0)
MODEL_PATH = 'full_model/lite_model.tflite'

# Global variables
exit_flag = threading.Event()
event_buffer = queue.Queue()
using_virtual_mic = False
audio_files_to_process = []

def setup_logging():
    """Set up logging with rotating file handlers and console output."""
    os.makedirs("logs", exist_ok=True)
    
    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)
    
    # Console handler with formatted output
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    main_logger.addHandler(console)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        "logs/sound_detection.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    main_logger.addHandler(file_handler)
    
    # Event logger (separate file)
    event_logger = logging.getLogger("events")
    event_logger.setLevel(logging.INFO)
    
    event_handler = logging.handlers.RotatingFileHandler(
        "logs/events.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    event_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    event_logger.addHandler(event_handler)
    
    return main_logger, event_logger

def validate_config():
    """Validate essential configuration settings."""
    valid = True
    
    if not DEVICE:
        logging.error("Device serial number not set in config")
        valid = False
    
    if not FOLDER:
        logging.error("Cache folder not set in config")
        valid = False
        
    if not BASE_API_URL:
        logging.warning("Base API URL not set - events will be stored locally only")
    
    if not os.path.exists(MODEL_PATH):
        logging.error(f"TFLite model file not found at {MODEL_PATH}")
        valid = False
    
    if RATE != 44100:
        logging.warning(f"Sample rate set to {RATE}Hz, but microphone requires 44100Hz")
    
    return valid

class BackoffStrategy:
    """Implements exponential backoff for retries."""
    def __init__(self, initial_delay=1, max_delay=60, factor=2):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.factor = factor
        self.current_delay = initial_delay
    
    def next_delay(self):
        """Get the next delay time and increase the internal counter."""
        delay = min(self.current_delay, self.max_delay)
        self.current_delay = self.current_delay * self.factor
        return delay
    
    def reset(self):
        """Reset the delay to the initial value."""
        self.current_delay = self.initial_delay

def truncate(num, n):
    """Truncate a float to n decimal places."""
    integer = int(num * (10**n))
    return float(integer / (10**n))

def load_data(filename, sample_rate=MODEL_RATE):
    """Load audio data from a WAV file and resample to 16000 Hz using librosa."""
    try:
        logging.info(f"Loading audio file: {filename}")
        data, rate = librosa.load(filename, sr=sample_rate, mono=True)
        if np.abs(data).max() > 0:
            data = data / np.abs(data).max()
        data = data.astype(np.float32)
        logging.debug(f"Audio shape: {data.shape}, Sample rate: {rate}")
        logging.debug(f"Value range: [{data.min():.3f}, {data.max():.3f}]")
        return data, rate
    except Exception as e:
        logging.error(f"Error loading audio data from {filename}: {e}")
        return None, None

def save_to_file(frames):
    """Save audio frames to a WAV file and return the filepath."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_recording.wav"
    filepath = os.path.join(FOLDER, filename)
    
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        logging.debug(f"Saved audio to {filepath}")
        return filepath, filename
    except Exception as e:
        logging.error(f"Error saving audio to {filepath}: {e}")
        return None, None

def analyze(filepath):
    """Analyze audio file using the TensorFlow Lite model."""
    data, sample_rate = load_data(filepath)
    if data is None:
        return None

    try:
        interpreter = Interpreter(MODEL_PATH)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logging.debug(f"Input details: {input_details}")
        logging.debug(f"Output details: {output_details}")
        
        input_data = np.array(data, dtype=np.float32)
        interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probabilities = tf.nn.softmax(output_data).numpy()
        
        top_index = np.argmax(probabilities)
        top_score = float(probabilities[top_index])
        
        categories = [
            "crying_baby", "sneezing", "clapping", "coughing", "footsteps",
            "laughing", "glass_breaking", "other", "gunshots", "human_speech",
            "bicycle", "bicycle_bell"
        ]
        
        result = {
            "classIndex": int(top_index),
            "className": categories[top_index],
            "confidenceScore": truncate(top_score, 4)
        }
        logging.info(f"Sound detected: {result['className']} with score {result['confidenceScore']}")
        return result
    except Exception as e:
        logging.error(f"Error analyzing audio: {e}")
        return None

def send_event_to_api(event_data):
    """Send event data to the API with retries."""
    event_logger = logging.getLogger("events")
    backoff = BackoffStrategy()
    
    if not BASE_API_URL:
        logging.warning("Base API URL is not set, cannot send events")
        return False
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            event_class = event_data["event_details"][0]["className"]
            confidence = event_data["event_details"][0]["confidenceScore"]
            location = event_data.get("location", "+0.0+0.0")
            timestamp = event_data["recording_start_time"]
            filepath = event_data["filepath"]
            
            event_logger.info(f"Event: {event_class}, Confidence: {confidence}, Location: {location}")
            logging.info(f"Sending event to API: {event_class}, Confidence: {confidence}")
            api_response = event_creation_IOT.create_event(
                name=f"Sound: {event_class}",
                category=event_class,
                id=DEVICE,
                location=location
            )
            
            if os.path.exists(filepath):
                logging.info(f"Uploading sound file: {filepath}")
                event_creation_IOT.upload_sound_file(
                    sound_file=filepath,
                    location=location,
                    key=api_response.get("id", "unknown")
                )
            
            logging.info(f"Event sent successfully: {event_class}")
            return True
        except Exception as e:
            retry_count += 1
            delay = backoff.next_delay()
            logging.warning(f"Error sending event to API (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Failed to send event after {max_retries} attempts")
                return False

def save_event_buffer():
    """Save the event buffer to a file for persistence."""
    buffer_file = os.path.join(FOLDER, "event_buffer.json")
    
    try:
        items = list(event_buffer.queue)
        serializable_items = []
        for item in items:
            serializable_item = item.copy()
            if "recording_start_time" in serializable_item:
                serializable_item["recording_start_time"] = serializable_item["recording_start_time"]
            serializable_items.append(serializable_item)
        
        with open(buffer_file, 'w') as f:
            json.dump(serializable_items, f)
        logging.info(f"Saved {len(items)} events to buffer file")
    except Exception as e:
        logging.error(f"Error saving event buffer: {e}")

def load_event_buffer():
    """Load the event buffer from a file."""
    buffer_file = os.path.join(FOLDER, "event_buffer.json")
    
    if not os.path.exists(buffer_file):
        logging.info("No event buffer file found, starting with empty buffer")
        return
    
    try:
        with open(buffer_file, 'r') as f:
            items = json.load(f)
        for item in items:
            event_buffer.put(item)
        logging.info(f"Loaded {len(items)} events from buffer file")
        os.remove(buffer_file)
    except Exception as e:
        logging.error(f"Error loading event buffer: {e}")

def process_event_buffer():
    """Process events in the buffer and send them to the API."""
    logging.info("Starting event buffer processor")
    backoff = BackoffStrategy(initial_delay=5)
    
    while not exit_flag.is_set():
        try:
            event_data = event_buffer.get(timeout=1)
            if send_event_to_api(event_data):
                backoff.reset()
                filepath = event_data.get("filepath")
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        logging.debug(f"Removed file after sending: {filepath}")
                    except Exception as e:
                        logging.error(f"Failed to remove file {filepath}: {e}")
            else:
                event_buffer.put(event_data)
                delay = backoff.next_delay()
                logging.warning(f"Failed to send event, will retry. Queue size: {event_buffer.qsize()}")
                time.sleep(delay)
            event_buffer.task_done()
        except queue.Empty:
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error in event buffer processor: {e}")
            time.sleep(5)
    
    logging.info("Event buffer processor stopping")
    save_event_buffer()

def process_audio_chunk(frames, actual_channels):
    """Process an audio chunk: save, analyze, and queue events if detected."""
    filepath, filename = save_to_file(frames)
    if not filepath:
        return
    
    result = analyze(filepath)
    if not result:
        try:
            os.remove(filepath)
            logging.debug(f"Removed file that couldn't be analyzed: {filepath}")
        except Exception as e:
            logging.error(f"Failed to remove file {filepath}: {e}")
        return
    
    detected_class = result["className"]
    confidence = result["confidenceScore"]
    confidence_percentage = confidence * 100
    
    # Terminal feedback
    print(f"\n[PREDICTION] {detected_class} - Confidence: {confidence_percentage:.1f}%")
    
    categories_of_interest = [
        "crying_baby", "sneezing", "clapping", "coughing",
        "glass_breaking", "gunshots", "bicycle_bell"
    ]
    
    if detected_class not in categories_of_interest or confidence < 0.6:
        try:
            os.remove(filepath)
            logging.debug(f"Removed file with no detected event: {filepath}")
        except Exception as e:
            logging.error(f"Failed to remove file {filepath}: {e}")
        return
    
    event_data = {
        "event_details": [result],
        "recording_duration": RECORD_SECONDS,
        "recording_start_time": datetime.datetime.now().isoformat(),
        "location": "+0.0+0.0",  # Static, no GPS
        "filename": filename,
        "filepath": filepath
    }
    event_buffer.put(event_data)
    print(f"[DETECTED] Added event to buffer: {detected_class} ({confidence_percentage:.1f}%)")
    logging.info(f"Added event to buffer: {detected_class}")



def record_audio():
    """Record audio continuously and process in chunks."""
    global using_virtual_mic
    print("\n[STARTING AUDIO RECORDING SYSTEM]")
    logging.info("Starting audio recording")
    
    audio_dir = Path("audio")
    if audio_dir.exists() and list(audio_dir.glob("*.wav")):
        if setup_virtual_mic():
            process_virtual_audio_files()
            return
        else:
            logging.error("Failed to set up virtual microphone")
    
    audio = pyaudio.PyAudio()
    stream = None
    actual_channels = CHANNELS
    
    # List devices for debugging
    print("\nAvailable PyAudio devices:")
    for i in range(audio.get_device_count()):
        dev_info = audio.get_device_info_by_index(i)
        print(f"Device {i}: {dev_info['name']} - {dev_info['maxInputChannels']} input channels")
    
    # Try preferred device
    try:
        dev_info = audio.get_device_info_by_index(PREFERRED_DEVICE_INDEX)
        if dev_info['maxInputChannels'] > 0:
            actual_channels = min(CHANNELS, int(dev_info['maxInputChannels']))
            stream = audio.open(
                format=FORMAT,
                channels=actual_channels,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=PREFERRED_DEVICE_INDEX
            )
            print(f"SUCCESS: Connected to USB mic at index {PREFERRED_DEVICE_INDEX}: {dev_info['name']}")
    except Exception as e:
        print(f"Failed to open preferred device {PREFERRED_DEVICE_INDEX}: {e}")
    
    # Fallback to USB device
    if not stream:
        for i in range(audio.get_device_count()):
            dev_info = audio.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0 and 'usb' in dev_info['name'].lower():
                actual_channels = min(CHANNELS, int(dev_info['maxInputChannels']))
                try:
                    stream = audio.open(
                        format=FORMAT,
                        channels=actual_channels,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=i
                    )
                    print(f"SUCCESS: Connected to fallback USB mic at index {i}: {dev_info['name']}")
                    break
                except Exception as e:
                    print(f"Failed to open device {i}: {e}")
    
    # Switch to virtual mic if no real mic works
    if not stream:
        if setup_virtual_mic():
            process_virtual_audio_files()
        else:
            audio.terminate()
            raise Exception("No audio input available")
        return
    
    print("\nRECORDING STARTED - Press Ctrl+C to stop")
    chunk_counter = 0
    
    while not exit_flag.is_set():
        chunk_counter += 1
        print(f"\n--- Recording Chunk #{chunk_counter} ---")
        frames = []
        
        for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
            if exit_flag.is_set():
                break
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                if i % 10 == 0:
                    progress = int(100 * i / (RATE / CHUNK * RECORD_SECONDS))
                    audio_sample = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    level = np.abs(audio_sample).mean()
                    bar_count = int(level * 30)
                    level_bar = "#" * bar_count + "-" * (30 - bar_count)
                    sys.stdout.write(f"\rRecording: {progress}% [{'=' * (progress//5)}>{' ' * (19-progress//5)}] Level: [{level_bar}]")
                    sys.stdout.flush()
            except Exception as e:
                print(f"\nERROR reading from stream: {e}")
                if setup_virtual_mic():
                    stream.stop_stream()
                    stream.close()
                    audio.terminate()
                    process_virtual_audio_files()
                    return
                break
        
        if len(frames) >= int(RATE / CHUNK * RECORD_SECONDS * 0.75):
            thread = threading.Thread(
                target=process_audio_chunk,
                args=(frames.copy(), actual_channels),
                name=f"Process-{datetime.datetime.now().strftime('%H%M%S')}"
            )
            thread.daemon = True
            thread.start()
        else:
            print(f"WARNING: Incomplete chunk ({len(frames)} frames), skipping")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("\nAudio recording stopped")
    logging.info("Audio recording stopped")

def signal_handler(sig, frame):
    """Handle termination signals gracefully."""
    print("\nSTOPPING - Please wait...")
    logging.info(f"Received signal {sig}, shutting down...")
    exit_flag.set()
    save_event_buffer()
    time.sleep(2)
    logging.info("Shutdown complete")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main_logger, event_logger = setup_logging()
    
    print("\n" + "=" * 60)
    print("SOUND DETECTION SYSTEM".center(60))
    print("=" * 60)
    
    if not validate_config():
        print("ERROR: Configuration validation failed, exiting")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"   * Audio window: {RECORD_SECONDS} seconds")
    print(f"   * Recording sample rate: {RATE}Hz")
    print(f"   * Model sample rate: {MODEL_RATE}Hz")
    print(f"   * Channels: {CHANNELS}")
    print(f"   * Preferred device index: {PREFERRED_DEVICE_INDEX}")
    
    os.makedirs(FOLDER, exist_ok=True)
    load_event_buffer()
    buffer_thread = threading.Thread(target=process_event_buffer, name="BufferProcessor", daemon=True)
    buffer_thread.start()
    record_audio()
    save_event_buffer()
    print("\nSound Detection System stopped")
    logging.info("Sound Detection System stopped")
