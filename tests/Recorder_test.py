import unittest
from src.Recorder import Recorder
import numpy as np
from os.path import exists

class TestRecorder(unittest.TestCase):
    def test_record(self):
        recorder = Recorder()
        audio_data = recorder.record(1)
        self.assertTrue(type(audio_data) is np.ndarray)
    
    def test_save(self):
        recorder = Recorder()
        audio_data = recorder.record(1)
        recorder.save('tests/test_recorder.wav')
        self.assertTrue(exists('tests/test_recorder.wav'))

    def test_play(self):
        recorder = Recorder()
        audio_data = recorder.record(1)
        recorder.play()