# Standard imports
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import joblib
from pprint import pprint
import unittest

class TestSanity(unittest.TestCase):
    model = None
    X_test = None
    y_test = None

    def setUp(self):
        # Load data and check
        data = pd.read_csv('data/iris.csv')

        # Create Train and Test splits
        train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
        X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
        y_train = train.species
        self.X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
        self.y_test = test.species

        self.model = joblib.load("models/dtree.joblib")

    def test_sanity(self):
        predictions = self.model.predict(self.X_test)
        accuracy    = metrics.accuracy_score(predictions, self.y_test)
        self.assertGreater(accuracy, 0.9, "Accuracy < 0.9")

if __name__ == '__main__':
    unittest.main()

