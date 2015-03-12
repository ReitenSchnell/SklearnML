from unittest import TestCase
import numpy
from sklearn.naive_bayes import GaussianNB

def predict():
    x = numpy.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = numpy.array([1, 1, 1, 2, 2, 2])
    clf = GaussianNB()
    clf.fit(x, y)
    result = clf.predict([0.8, 1])
    return result

class PredictTests(TestCase):
    def test_prediction(self):
        result = predict()
        self.assertEqual(result, 2)