import unittest
from src.ResultsManager import ResultsManager
import pandas as pd
from os.path import exists

class TestResultsManager(unittest.TestCase):
    results_manager = ResultsManager('results_test.csv')
    dummy_results = {'MDVP:Fo(Hz)': 99.078, 'MDVP:Fhi(Hz)': 101.045, 'MDVP:Flo(Hz)': 95.789, 'MDVP:Jitter(%)': 0.0057799999999999995, 'MDVP:Jitter(Abs)': 5.8296e-05, 'MDVP:RAP': 0.0034300000000000003, 'MDVP:PPQ': 0.00345, 'Jitter:DDP': 0.010289999999999999, 'MDVP:Shimmer': 0.022639999999999997, 'MDVP:Shimmer(dB)': 0.211, 'Shimmer:APQ3': 0.01098, 'Shimmer:APQ5': 0.01307, 'MDVP:APQ': 0.0181, 'Shimmer:DDA': 0.03293, 'NHR': 0.014735, 'HNR': 20.432}

    def test_init(self):
        no_results = ResultsManager('randomfile')
        self.assertEqual(no_results.path, 'randomfile.csv')
        self.assertEqual(no_results.results, {})

        results = ResultsManager('results_test.csv')
        self.assertTrue('454' in results.results)
        self.assertEqual(results.path, 'results_test.csv')
    
    def test_add_results(self):
        uid = self.results_manager.add_results(self.dummy_results)
        self.assertTrue(uid in self.results_manager.results)
        for feature, value in self.dummy_results.items():
            self.assertTrue((feature,value) in self.results_manager.results[uid].items())

    def test_get_status(self):
        self.assertEqual(self.results_manager.get_status('454'), 1)
        self.assertEqual(self.results_manager.get_status('659'), 0)
        self.assertEqual(self.results_manager.get_status('335'), None)

    def test_get_features(self):
        features = self.results_manager.get_features('725')
        features = {k:v for k, v in features.items() if v is not None} # Remove null values
        self.assertEqual(features, self.dummy_results)

    def test_set_status(self):
        self.results_manager.set_status('335', 1)
        self.assertEqual(self.results_manager.results['335']['status'], 1)
        self.results_manager.set_status('335', 0)
        self.assertEqual(self.results_manager.results['335']['status'], 0)

    def test_remove_results(self):
        self.assertTrue('725' in self.results_manager.results)
        self.results_manager.remove_results('725')
        self.assertFalse('725' in self.results_manager.results)

    def test_get_unpredicted(self):
        self.assertEqual(['335','351','385'], self.results_manager.get_unpredicted())

    def test_to_dataframe(self):
        df = self.results_manager.to_dataframe()
        self.assertEqual(type(df), pd.DataFrame)
        self.assertTrue('335' in df.index)

    def test_save(self):
        self.results_manager.save('new_results.csv')
        self.assertTrue(exists('new_results.csv'))
        df = pd.read_csv('new_results.csv', index_col=0).to_dict(orient='index')
        df = {str(k): v for k, v in df.items()}
        self.assertTrue('335' in df)
