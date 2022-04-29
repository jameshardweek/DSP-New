import wave

import numpy as np
import sounddevice as sd


class Recorder:
    def __init__(self) -> None:
        self.SAMPLERATE = 44100
        self.CHANNELS = 1

    # Record audio data from user's microphone for specified length
    def record(self, length):
        print(f"Recording for {length} seconds...")
        audio = sd.rec(int(length * self.SAMPLERATE), samplerate=self.SAMPLERATE, channels=self.CHANNELS, blocking=True)
        # Normalise audio to 16-bit range
        audio = audio / audio.max() * np.iinfo(np.int16).max
        audio = audio.astype(np.int16)
        self.audio_data = audio
        print(f"Recording complete.")
        return audio

    # Save audio data to file
    def save(self, filename: str):
        if self.audio_data is not None:
            with wave.open(filename, 'wb') as f:
                f.setnchannels(1)
                f.setframerate(44100)
                f.setsampwidth(2)
                f.writeframes(self.audio_data)

    # Playback recorded audio
    def play(self):
        if self.audio_data is not None:
            sd.play(self.audio_data)
