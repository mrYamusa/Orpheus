import librosa
import numpy as np
from vibenet import load_model
import asyncio

# Load the pre-trained EfficientNet model
model = load_model()

song_path = "./LilUziVert_WhatYouSaying.mp3"

async def analyze_song(path):
    # --- STAGE 1: Data Loading & VibeNet ---
    print("Stage 1: Loading Audio & Running VibeNet...")
    async with asyncio.TaskGroup() as tg:
        # Schedule VibeNet
        task_vibenet = tg.create_task(asyncio.to_thread(model.predict, path))
        # Schedule Audio Loading
        task_audio = tg.create_task(asyncio.to_thread(librosa.load, path, sr=22050))
    
    # Get results from Stage 1
    vibe_result = task_vibenet.result()
    y, sr = task_audio.result()

    # --- STAGE 2: Feature Extraction (Requires 'y' and 'sr') ---
    print("Stage 2: Extracting Tone, Harmony, and Rhythm...")
    async with asyncio.TaskGroup() as tg:
        # 1. Tempo (Rhythm)
        task_tempo = tg.create_task(asyncio.to_thread(librosa.beat.beat_track, y=y, sr=sr))
        
        # 2. RMS (Energy/Loudness)
        task_rms = tg.create_task(asyncio.to_thread(librosa.feature.rms, y=y))
        
        # 3. Spectral Centroid (Tone/Brightness)
        # Higher = Brighter (Pop/Trap), Lower = Darker (Lo-fi/Bass)
        task_centroid = tg.create_task(asyncio.to_thread(librosa.feature.spectral_centroid, y=y, sr=sr))
        
        # 4. Chroma (Harmony/Key Profile)
        # Extracts the 12 pitch classes (C, C#, D...) to see harmonic content
        task_chroma = tg.create_task(asyncio.to_thread(librosa.feature.chroma_stft, y=y, sr=sr))

    # Return everything neatly
    return (
        vibe_result[0],  # The VibeNet result
        task_tempo.result(),
        task_rms.result(),
        task_centroid.result(),
        task_chroma.result()
    )

# Run the async analysis
print("Starting analysis...")
vibe, tempo_data, rms_data, centroid_data, chroma_data = asyncio.run(analyze_song(song_path))

# --- PRINT RESULTS ---

print(f"\nResults for 'What You Saying':")
print("-" * 30)

# 1. VibeNet Stats
print(f"Valence (Sad -> Happy): {vibe.valence:.2f}")
print(f"Energy (Chill -> Hype): {vibe.energy:.2f}")
print(f"Danceability:           {vibe.danceability:.2f}")

# 2. Rhythm
print("-" * 30)
# print(f"Estimated Tempo:        {tempo_data[0]:.2f} BPM")
print(f"Estimated Tempo:        {tempo_data[0]} BPM")

# 3. Physical Energy (Loudness)
avg_energy = np.mean(rms_data)
print(f"Relative Energy (RMS):  {avg_energy:.4f}")

# 4. Tone (Spectral Centroid)
# We take the mean to get the average "brightness" of the whole song
avg_brightness = np.mean(centroid_data)
tone_type = "Bright/Sharp" if avg_brightness > 2000 else "Dark/Mellow"
print("-" * 30)
print(f"Tone Brightness:        {avg_brightness:.2f} Hz")
print(f"Perceived Tone:         {tone_type}")

# 5. Harmony (Chroma)
# This is a 12xTime matrix. We take the mean to see which notes are most dominant.
avg_chroma = np.mean(chroma_data, axis=1)
# Map indices to notes
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
dominant_note_index = np.argmax(avg_chroma)
print(f"Dominant Note/Key Center: {notes[dominant_note_index]}")


"""# Initialize the MusicExtractor
# We limit stats to 'mean' to get a single value per track
features, features_frames = es.MusicExtractor(
    lowlevelStats=['mean'], 
    rhythmStats=['mean'], 
    tonalStats=['mean']
)('./LilUziVert_WhatYouSaying.mp3')

# 1. Tempo (BPM)
bpm = features['rhythm.bpm']

# 2. Danceability
# Note: This is an estimation of how danceable the rh
# ythm is
danceability = features['rhythm.danceability']

print(f"Tempo: {bpm:.2f} BPM")
print(f"Danceability: {danceability:.2f}")
print("\nAll extracted features:")
for key, value in features.items():
    print(f"{key}: {value}")
"""


"""smaple_rate = librosa.get_samplerate("./LilUziVert_WhatYouSaying.mp3")
print(f"{smaple_rate}")

y, sr = librosa.load("./LilUziVert_WhatYouSaying.mp3", sr=22050)

tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(f"Tempo: {tempo} BPM")
print(f"Beat frames: {beat_frames}")


rms = librosa.feature.rms(y=y)
energy_proxy = np.mean(rms)"""
"""
stream = librosa.stream("./LilUziVert_WhatYouSaying.mp3", block_length=2048, frame_length=2048, hop_length=512, fill_value=0, dtype=np.float32)

for y_block in stream:
    D_block = librosa.stft(y_block, center=False)
    print(f"{D_block.shape} \n{D_block}")"""