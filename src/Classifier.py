import pandas as pd
from sklearn import svm
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from joblib import load, dump
from os.path import exists
import numpy as np


class Classifier:
    # def __init__(self) -> None:
        # dataset = pd.read_csv('data/parkinsons.data')
        # y = dataset['status']
        # X = dataset.drop(['name', 'status', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'], axis=1)

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # if type is None or type == 'svm':
        #     self.clf = svm.SVC(probability=True)
        # else:
        #     if type == 'rf':
        #         self.clf = RandomForestClassifier()
        #     elif type == 'ada':
        #         self.clf = AdaBoostClassifier()
        
        # self.train(self.X_train, self.y_train)
        # pass
        
    def train(self, X, y) -> None:
        self.clf = self.clf.fit(X, y)
        return self.clf
        # return self.clf

    def cross_validate(self, X, y) -> float:
        scores_test = cross_val_score(self.clf, X.values, y.values, scoring='accuracy', cv=10)
        return np.mean(scores_test)

    # def cross_validate(self, X, y):
    #     scores_test = cross_val_score(self.clf, X.values, y.values, scoring='accuracy', cv=10)
    #     return np.mean(scores_test)

    def predict(self, features) -> list:
        # to_predict = []
        # for key, value in features.items():
        #     to_predict.append(value)
        return self.clf.predict(features)

    def predict_probability(self, features) -> list:
        return self.clf.predict_proba(features)

    def confusion_matrix(self, X, y) -> ConfusionMatrixDisplay:
        return ConfusionMatrixDisplay.from_estimator(self.clf, X, y)

    def classification_report(self, X, y) -> str:
        predictions = self.clf.predict(X)
        return(classification_report(y, predictions))

    def save(self) -> None:
        dump(self.clf, "models/" + type(self).__name__)
        print(f"Model saved to 'models/{type(self).__name__}'.")

    def load(self, file):
        if exists(file):
            try:
                clf = load(file)
                if type(clf).__name__ == type(self.clf).__name__:
                    self.clf = clf
                    return True
                else:
                    # print(f"Selected model type differs from current instance. Model not loaded.")
                    print(f"Attempted to load {type(clf).__name__} when current classifier is {type(self).__name__}")
                    print(f"Model not loaded.")
                    return False
            except:
                print(f"Loading classifier failed.")
                return False

class SVM(Classifier):
    def __init__(self, params=None) -> None:
        if params is None:
            self.clf = svm.SVC(
                probability=True,
            )
        elif type(params) is dict:
            try:
                self.clf = svm.SVC(
                    probability=True,
                    C = params['C'] if 'C' in params else 1,
                    kernel = params['kernel'] if 'kernel' in params else 'rbf',
                    gamma = params['gamma'] if 'gamma' in params else 'scale',
                    random_state=params['random_state'] if 'random_state' in params else None,
                )
            except:
                self.__init__()
        else:
            self.__init__()

class RandomForest(Classifier):
    def __init__(self, params=None) -> None:
        if params is None:
            self.clf = RandomForestClassifier()
        elif type(params) is dict:
            try:
                self.clf = RandomForestClassifier(
                    n_estimators=params['n_estimators'] if 'n_estimators' in params else 100,
                    criterion=params['criterion'] if 'criterion' in params else 'gini',
                    max_depth=params['max_depth'] if 'max_depth' in params else None,
                    min_samples_split=params['min_samples_split'] if 'min_samples_split' in params else 2,
                    max_leaf_nodes=params['max_leaf_nodes'] if 'max_leaf_nodes' in params else None,
                    max_features=params['max_features'] if 'max_features' in params else 'auto',
                    max_samples=params['max_samples'] if 'max_samples' in params else None,
                    random_state=params['random_state'] if 'random_state' in params else None,
                )
            except:
                self.__init__()
        else:
            self.__init__()

class AdaBoost(Classifier):
    def __init__(self, params=None) -> None:
        if params is None:
            self.clf = AdaBoostClassifier()
        elif type(params) is dict:
            try:
                self.clf = AdaBoostClassifier(
                    n_estimators=params['n_estimators'] if 'n_estimators' in params else 50,
                    learning_rate=params['learning_rate'] if 'learning_rate' in params else 1,
                    algorithm=params['algorithm'] if 'algorithm' in params else 'SAMME.R',
                    random_state=params['random_state'] if 'random_state' in params else None,
                )
            except:
                self.__init__()
        else:
            self.__init__()

class Voting(Classifier):
    def __init__(self, *args) -> None:
        try:
            estimators = []
            for i, clf in enumerate(args):
                estimators.append((f'{i}', clf.clf))

            self.clf = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
            )
        except Exception as e:
            print(f"Init failed. {e}")
            return None

def main():
    dataset_url = "data/parkinsons.data"
    df = pd.read_csv(dataset_url)
    y = df['status']
    X = df.drop(['name', 'status', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    svm = SVM()
    svm.train(X_train, y_train)
    rf = RandomForest()
    rf.train(X_train, y_train)
    ada = AdaBoost()
    ada.train(X_train, y_train)

    # print(svm.cross(X_test, y_test))
    # print(rf.test(X_test, y_test))
    # print(ada.test(X_test, y_test))

    vc = Voting(svm,rf,ada)
    vc.train(X_train, y_train)
    # print(vc.test(X_test, y_test))

    svm.save()
    rf.save()
    ada.save()
    vc.save()
    
    print()

if __name__ == '__main__':
    main()