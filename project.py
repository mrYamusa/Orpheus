import librosa
import numpy as np
import asyncio
from vibenet import load_model

# Load model
model = load_model()
song_path = "./LilUziVert_WhatYouSaying.mp3"

async def analyze_frames(path):
    # 1. Load Audio
    # We use a standard hop_length of 512 samples (~23ms per frame)
    y, sr = await asyncio.to_thread(librosa.load, path, sr=22050)
    
    # 2. Extract Features Over Time (Returns arrays, not single numbers)
    # Energy (RMS) over time
    rms = await asyncio.to_thread(librosa.feature.rms, y=y)
    
    # Brightness (Spectral Centroid) over time
    centroid = await asyncio.to_thread(librosa.feature.spectral_centroid, y=y, sr=sr)
    
    # 3. Create a Time Grid
    # This gives us the timestamp (in seconds) for every single frame
    times = librosa.times_like(rms, sr=sr)
    
    return times, rms[0], centroid[0]

def find_significant_events(times, values, feature_name, threshold_factor=1.5):
    """
    Scans the timeline and returns moments that are 'threshold_factor' times 
    higher than the average (Outliers).
    """
    avg_val = np.mean(values)
    std_dev = np.std(values)
    
    # Define an "Outlier" as anything X standard deviations above the mean
    # You can tweak this. Higher = strictly only the biggest peaks.
    threshold = avg_val + (std_dev * threshold_factor)
    
    print(f"\n--- Significant {feature_name} Events (Threshold: {threshold:.2f}) ---")
    
    # We group consecutive frames to avoid printing 10 lines for 1 second of audio
    events = []
    currently_peaking = False
    
    for i, val in enumerate(values):
        if val > threshold:
            if not currently_peaking:
                # Start of a new peak event
                print(f"Time: {times[i]:.2f}s | Value: {val:.2f}")
                currently_peaking = True
        else:
            currently_peaking = False

# --- RUN ANALYSIS ---

# 1. Get the raw timeline data
times, energy_over_time, brightness_over_time = asyncio.run(analyze_frames(song_path))

# 2. Detect Energy Spikes (e.g., Beat drops, loud vocals)
# We look for moments 2 standard deviations louder than average
find_significant_events(times, energy_over_time, "Loudness (Energy)", threshold_factor=2.0)

# 3. Detect Tone Spikes (e.g., High pitch screeches, cymbal crashes)
# We look for moments where the sound gets very "bright" (high frequency)
find_significant_events(times, brightness_over_time, "Brightness (Frequency)", threshold_factor=2.0)

# 4. (Optional) Accessing specific raw frames
# If you want to know exactly what happened at the 10-second mark:
target_time = 10.0
# Find the frame index closest to 10 seconds
frame_index = np.argmin(np.abs(times - target_time))

print(f"\n--- Snapshot at {target_time} seconds ---")
print(f"Exact Time: {times[frame_index]:.2f}s")
print(f"Energy:     {energy_over_time[frame_index]:.4f}")
print(f"Brightness: {brightness_over_time[frame_index]:.2f} Hz")