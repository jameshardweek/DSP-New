import sounddevice as sd
import wave
import numpy as np

class Recorder:
    def __init__(self) -> None:
        self.SAMPLERATE = 44100
        self.CHANNELS = 1

    # Record audio data from user's microphone for specified length
    def record(self, length):
        print(f"Recording for {length} seconds...")
        audio = sd.rec(int(length * self.SAMPLERATE), samplerate=self.SAMPLERATE, channels=self.CHANNELS, blocking=True)
        audio = audio / audio.max() * np.iinfo(np.int16).max
        audio = audio.astype(np.int16)
        self.audio_data = audio
        print(f"Recording complete.")
        return audio

    # Save audio data to file
    def save(self, filename: str):
        with wave.open(filename + '.wav', 'wb') as f:
            f.setnchannels(1)
            f.setframerate(44100)
            f.setsampwidth(2)
            f.writeframes(self.audio_data)

    # Playback recorded audio
    def play(self):
        sd.play(self.audio_data)