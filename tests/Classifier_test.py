from random import Random
import unittest

from sklearn.exceptions import NotFittedError
from src.Classifier import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from os.path import exists
import os
from joblib import load

class TestClassifier(unittest.TestCase):
    dataset = pd.read_csv('data/parkinsons.data')
    y = dataset['status']
    X = dataset.drop(['name', 'status', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Test initialisaiton of each of the classifier types, and ensure that parameters can be passed
    def test_init(self):
        svm_test_params = {
            'C': 5,
            'gamma': 0.5,
            'kernel': 'poly',
            'invalid_param': 19,
        }

        svm = SVM(svm_test_params)
        svm_actual_params = svm.clf.get_params()

        for k, v in svm_actual_params.items():
            if k in svm_test_params:
                self.assertEqual(svm_test_params[k], v)

        ada_test_params = {
            'n_estimators': 10,
            'learning_rate': 0.1,
            'invalid_param': 19,
        }

        ada = AdaBoost(ada_test_params)
        ada_actual_params = ada.clf.get_params()

        for k, v in ada_actual_params.items():
            if k in ada_test_params:
                self.assertEqual(ada_test_params[k], v)

        rf_test_params = {
            'n_estimators': 10,
            'criterion': 'entropy',
            'max_depth': 5,
            'min_samples_split': 0.5,
            'invalid_param': 19,
        }

        rf = RandomForest(rf_test_params)
        rf_actual_params = rf.clf.get_params()

        for k, v in rf_actual_params.items():
            if k in rf_test_params:
                self.assertEqual(rf_test_params[k], v)

        vc_test_estimators = [svm.clf, ada.clf, rf.clf]
        vc = Voting(svm, ada, rf)
        vc_actual_params = vc.clf.get_params()

        for i, estimator in enumerate(vc_actual_params['estimators']):
            self.assertTrue(type(estimator[1]) == type(vc_test_estimators[i]))

    # Ensure that the classifier can be trained, provided training data
    def test_train(self):
        svm = SVM()

        # Should throw NotFittedError
        with self.assertRaises(NotFittedError):
            check_is_fitted(svm.clf)

        svm.train(self.X_train, self.y_train)
        
        # Should not throw NotFittedError
        try:
            check_is_fitted(svm.clf)
        except NotFittedError:
            self.fail()

    # Ensure that the classifier achieves the intended accuracy including unseen data
    def test_cross_validate(self):
        svm = SVM()
        svm.train(self.X_train, self.y_train)

        ada = AdaBoost()
        ada.train(self.X_train, self.y_train)

        rf = RandomForest()
        rf.train(self.X_train, self.y_train)

        vc = Voting(svm, ada, rf)
        vc.train(self.X_train, self.y_train)

        score = vc.cross_validate(self.X, self.y)
        # Ensure score including unseen data is greater than 70%
        self.assertGreater(score, 0.7)

    # Test the predict function on a arbitrary set of features in the dataset
    def test_predict(self):
        test_features = self.X.iloc[0]
        test_status = self.y.iloc[0]

        svm = SVM()
        svm.train(self.X_train, self.y_train)

        prediction = svm.predict([test_features])[0]

        self.assertEqual(test_status, prediction)

    # Ensure classifier confidence is above 70%
    def test_predict_probability(self):
        test_features = self.X.iloc[0]
        test_status = self.y.iloc[0]

        svm = SVM()
        svm.train(self.X_train, self.y_train)

        prediction = svm.predict_probability([test_features])[0][1]

        self.assertGreater(prediction, 0.7)

    # Ensure that this function returns a ConfusionMatrixDisplay
    def test_confusion_matrix(self):
        svm = SVM()
        svm.train(self.X_train, self.y_train)

        confusion_matrix = svm.confusion_matrix(self.X_test, self.y_test)

        self.assertEqual(type(confusion_matrix), ConfusionMatrixDisplay)

    # Ensure that a classifcation report can be generated
    def test_classification_report(self):
        svm = SVM()
        svm.train(self.X_train, self.y_train)
        report = svm.classification_report(self.X_test, self.y_test)
        self.assertIsNotNone(report)
        self.assertEqual(type(report), str)

    # Test saving the classifier to a file, and that when loaded, data is the same
    def test_save(self):
        svm = SVM()
        svm.train(self.X_train, self.y_train)

        os.remove('models/SVM')

        svm.save()
        self.assertTrue(exists('models/SVM'))

        loaded_svm = load('models/SVM')
        loaded_svm.score(self.X_test, self.y_test)

        self.assertIsNotNone(loaded_svm)

    # Ensure that a classifier can be loaded
    # Also makes sure that classifiers cannot load an incorrect classifier type
    def test_load(self):
        svm = SVM()

        with self.assertRaises(NotFittedError):
            check_is_fitted(svm.clf)

        svm.load('models/SVM')

        try:
            check_is_fitted(svm.clf)
        except NotFittedError:
            self.fail()

        rf = RandomForest()
        with self.assertRaises(NotFittedError):
            check_is_fitted(rf.clf)

        self.assertFalse(rf.load('models/SVM'))
