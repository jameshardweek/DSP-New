import pandas as pd
from sklearn import svm
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split


class Classifier:
    def __init__(self, type=None, params=None) -> None:
        dataset = pd.read_csv('data/parkinsons.data')
        y = dataset['status']
        X = dataset.drop(['name', 'status', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        if type is None or type == 'svm':
            self.clf = svm.SVC(probability=True)
        else:
            if type == 'rf':
                self.clf = RandomForestClassifier()
            elif type == 'ada':
                self.clf = AdaBoostClassifier()
        
        self.train(self.X_train, self.y_train)
        
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        self.clf = self.clf.fit(X, y)

    def test(self, X, y):
        self.X_test = X
        self.y_test = y
        return self.clf.score(X, y)

    def predict(self, features):
        # to_predict = []
        # for key, value in features.items():
        #     to_predict.append(value)
        return self.clf.predict(features)

    def get_probabilities(self, features):
        return self.clf.predict_proba(features)

    def confusion_matrix(self, X_test=None, y_test=None):
        return ConfusionMatrixDisplay.from_estimator(self.clf, self.X_test, self.y_test)

    def classification_report(self, X_test, y_test):
        predictions = self.clf.predict(X_test)
        return(classification_report(y_test, predictions))

class SVM(Classifier):
    def __init__(self, type=None) -> None:
        super().__init__('svm')

class RandomForest(Classifier):
    def __init__(self, type=None) -> None:
        super().__init__('rf')

class AdaBoost(Classifier):
    def __init__(self, type=None) -> None:
        super().__init__('ada')

class VotingClassifier(Classifier):
    def __init__(self, estimators=None) -> None:
        if estimators is not None:
            self.clf = VotingClassifier(
                estimators=estimators,
                voting='soft'
            )
        else:
            self.clf = VotingClassifier(
                voting='soft'
            )
