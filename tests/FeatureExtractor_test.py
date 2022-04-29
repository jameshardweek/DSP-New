import unittest
import pandas as pd

from src.FeatureExtractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    sound_file = 'tests/test.wav'
    dataset_labels = dataset_labels = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3",
        "Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"
    ]

    def test_init(self):
        feature_extractor = FeatureExtractor(self.sound_file)
        self.assertTrue(feature_extractor.features != {})
        for feature, value in feature_extractor.features.items():
            if value is not None:
                self.assertTrue(type(value) == float)

    def test_get_features(self):
        feature_extractor = FeatureExtractor(self.sound_file)
        features = feature_extractor.get_features()
        self.assertEqual(features, feature_extractor.features)
        # Values of these should be the same as they represent the same feature
        selected_features = feature_extractor.get_features(['Mean pitch', 'MDVP:Fo(Hz)']) 
        self.assertEqual(selected_features['Mean pitch'], selected_features['MDVP:Fo(Hz)'])
        some_null_features = feature_extractor.get_features(['hehekahsf','iaw','Mean pitch'])
        self.assertTrue(some_null_features['Mean pitch'])

    def test_get_dataset_features(self):
        feature_extractor = FeatureExtractor(self.sound_file)
        features = feature_extractor.get_dataset_features()
        for label in self.dataset_labels:
            self.assertTrue(label in features)

    def test_to_dataframe(self):
        feature_extractor = FeatureExtractor(self.sound_file)
        df = feature_extractor.to_dataframe()
        self.assertEqual(type(df), pd.DataFrame)
        for label in self.dataset_labels:
            self.assertTrue(label in df.columns)