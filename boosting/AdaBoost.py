__author__ = 'jiachiliu'

import numpy as np
from validation import confusion_matrix, confusion_matrix_analysis
from ranking import auc, roc
import random


class Predictor:
    """
    A predictor is used to predict the label of given data point,
    it can be the function the fitted by linear regression or
    root of the decision tree, or some other type of decision stump
    """

    def __init__(self):
        pass

    def predict_single(self, data_point):
        raise NotImplementedError("Abstract class does not implement this method")

    def predict(self, test):
        raise NotImplementedError("Abstract class does not implement this method")


class FeatureThresholdPredictor(Predictor):
    """
    A feature-threshold predictor is just a a pair with feature index and threshold value.
    Given an input instance to classify, a FeatureThresholdPredictor corresponding to feature f and threshold t,
    will predict +1 if the input instance has a feature f value exceeding the threshold t; otherwise, it predicts -1.
    """

    def __init__(self, feature, threshold):
        Predictor.__init__(self)
        self.feature = feature
        self.threshold = threshold
        # a boolean list indicate whether a data point get a wrong answer
        self.predicts = None

    def predict_single(self, data_point):
        """
        Predict the label of given data point
        :param data_point: a test data point without label
        :return: the label of test data point, +1 of -1
        """
        return 1.0 if data_point[self.feature] > self.threshold else -1.0

    def predict(self, test):
        """
        Predict the label for each data in given test data set
        :param test: test data set
        :return: a np.array contains all labels of the test data set
        """
        return np.array([self.predict_single(t) for t in test])

    def set_predicts(self, predict_labels, actual_labels):
        self.predicts = predict_labels != actual_labels


class DiscreteFeatureThresholdPredictor(FeatureThresholdPredictor):
    def __init__(self, feature, threshold):
        FeatureThresholdPredictor.__init__(self, feature, threshold)

    def predict_single(self, data_point):
        """
        Predict the label of given data point
        :param data_point: a test data point without label
        :return: the label of test data point, +1 of -1
        """
        return 1.0 if data_point[self.feature] == self.threshold else -1.0


class WeakLearner:
    """
    A weak leaner will take train dataset and distribution of each data point in dataset
    and returns a suitable predictor based on strategy.
    """

    def __init__(self):
        pass

    def setup_predictors(self, train):
        raise NotImplementedError("Abstract class does not implement this method")

    def fit(self, train, target, dist):
        raise NotImplementedError("Abstract class does not implement this method")


class OptimalWeakLearner(WeakLearner):
    """
    A optimal weak learner returns the "best" decision stump with respect to the weighted training set given.
    Here, the "best" decision stump h is the one whose error is as far from 1/2 as possible; in other words,
    the goal is to maximize abs(0.5 - error(h))
    """

    def __init__(self):
        WeakLearner.__init__(self)
        self.predictors = []

    def setup_predictors(self, train, discrete_features=None):
        """
        Given a train dataset, setup all possible feature-threshold pair as predictors
        :param train: train data set
        """
        if not discrete_features:
            discrete_features = []
        print "setup_predictors..."
        m, n = train.shape
        for f in range(n):
            if f not in discrete_features:
                self.generate_predictors_on_numeric_feature(f, train[:, f])
            else:
                self.generate_predictors_on_discrete_feature(f, train[:, f])

    def generate_predictors_on_discrete_feature(self, f, column):
        """
        Given a column, generate all possible threshold for this column.
        :param f: feature index
        :param column: all values under the feature
        """
        for v in np.unique(column):
            self.predictors.append(DiscreteFeatureThresholdPredictor(f, v))

    def generate_predictors_on_numeric_feature(self, f, column):
        """
        Given a column, generate all possible threshold for this column
        :param f: feature index
        :param column: a list of values for a feature
        """
        unique_values = np.unique(column)
        unique_values = np.sort(unique_values)
        min_value = unique_values.min()
        max_value = unique_values.max()

        self.predictors.append(FeatureThresholdPredictor(f, min_value - 0.5))
        if len(unique_values) > 1:
            for i in range(1, len(unique_values)):
                p = FeatureThresholdPredictor(f, (unique_values[i - 1] + unique_values[i]) / 2.0)
                self.predictors.append(p)
        self.predictors.append(FeatureThresholdPredictor(f, max_value + 0.5))

    def fit(self, train, target, dist):
        """
        Fit the train set to get optimal predictor based on given distribution
        :param train: train data set
        :param target: labels of train data set
        :param dist: weights of each data point
        :return: a predictor that maximum abs(0.5 - error(h))
        """
        max_err = -1.0
        weighted_error = 0
        best_predictor = None
        for predictor in self.predictors:
            if predictor.predicts is None:
                hypothesis = predictor.predict(train)
                predictor.set_predicts(hypothesis, target)
            predicts = predictor.predicts
            error_h = dist[predicts].sum()
            error = abs(0.5 - error_h)
            if max_err < error:
                max_err = error
                weighted_error = error_h
                best_predictor = predictor
        return best_predictor, weighted_error


class RandomChooseLeaner(OptimalWeakLearner):
    """
    A random choose learner will randomly choose a predictor in its
    predictor collection.
    """

    def __init__(self):
        OptimalWeakLearner.__init__(self)

    def fit(self, train, target, dist):
        """
        Fit the train data with random selected predictor
        :param train: train data set
        :param target: labels of train data set
        :param dist: weights of each data point
        :return: a random predictor with its weighted error
        """
        predictor = self.predictors[random.randint(0, len(self.predictors) - 1)]
        if predictor.predicts is None:
            hypothesis = predictor.predict(train)
            predictor.set_predicts(hypothesis, target)
        predicts = predictor.predicts
        error_h = dist[predicts].sum()
        return predictor, error_h


class AdaBoost:
    """
    A implementation of AdaBoost algorithm
    """

    def __init__(self, learner):
        self.learner = learner
        self.predictors = []
        self.confidence = []

    def boost(self, train, train_target, test, test_target, T=100, converged=0.001, discrete_features=None):
        """
        Running AdaBoost on given data set
        :param train: train data set
        :param train_target: train labels
        :param test: test data set
        :param test_target: test labels
        :param T: maximum of iteration
        :param converged: converge value
        """
        m, n = train.shape
        weights = np.array([1.0 / m] * m)
        self.learner.setup_predictors(train, discrete_features)

        # final hypothesis
        train_predicts = np.zeros(m)
        test_predicts = np.zeros(len(test))

        final_acc = 0
        final_auc = 0
        final_err = 0

        for t in range(T):
            predictor, weighted_err = self.learner.fit(train, train_target, weights)
            confidence = 0.5 * np.log((1.0 - weighted_err) / weighted_err)
            self.predictors.append(predictor)
            self.confidence.append(confidence)

            # accumulate final hypothesis on train
            predicts = predictor.predict(train)
            train_predicts += confidence * predicts
            train_predicts_signed = self.sign(train_predicts)
            train_cm = confusion_matrix(train_predicts_signed, train_target, 1.0, -1.0)
            train_err, train_acc, train_fpr, train_tpr = confusion_matrix_analysis(train_cm)

            # accumulate final hypothesis on test
            test_predicts += confidence * predictor.predict(test)
            test_predicts_signed = self.sign(test_predicts)
            test_cm = confusion_matrix(test_predicts_signed, test_target, 1.0, -1.0)
            test_err, test_acc, test_fpr, test_tpr = confusion_matrix_analysis(test_cm)
            roc_points = roc(test_target, test_predicts, 1.0, -1.0)
            test_auc = auc(roc_points[:, 1], roc_points[:, 0])

            final_acc = test_acc
            final_err = test_err
            final_auc = test_auc

            print "iteration %s: feature %s, threshold %s, round_error %s, train_error: %s, test_error: %s, auc: %s" % (
                t, predictor.feature, predictor.threshold, weighted_err, train_err, test_err, test_auc)

            # update weights
            for w in range(len(weights)):
                tmp = np.sqrt(weighted_err / (1.0 - weighted_err))
                if train_target[w] != predicts[w]:
                    tmp = np.sqrt((1.0 - weighted_err) / weighted_err)
                weights[w] = (weights[w] * tmp) / (2.0 * np.sqrt(weighted_err * (1 - weighted_err)))

        return final_acc, final_err, final_auc

    def hypothesis(self, test):
        """
        Given a test data set, get the final hypothesis for each data
        :param test: test data set
        :return: hypothesis score for test data set
        """
        h = np.zeros(test.shape[0])
        for i in range(len(self.confidence)):
            h += self.confidence[i] * self.predictors[i].predict(test)
        return h

    @staticmethod
    def sign(vals):
        """
        map every value in given value to +1 or -1. If the value is negative or zero map to -1
        otherwise map to +1
        :param vals: a list of value
        :return: a list of -1 or +1 based on the value
        """
        res = []
        for v in vals:
            if v <= 0:
                res.append(-1.0)
            else:
                res.append(1.0)
        return np.array(res)