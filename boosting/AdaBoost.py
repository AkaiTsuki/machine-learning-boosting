__author__ = 'jiachiliu'

import numpy as np


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
        self.predictors = {}
        self.predict_results = {}

    def setup_predictors(self, train):
        """
        Given a train dataset, setup all possible feature-threshold pair as predictors
        :param train: train data set
        """
        print "setup_predictors..."
        m, n = train.shape
        for f in range(n):
            self.predictors[f] = self.generate_predictors_on_feature(train[:, f])
        print self.predictors

    @staticmethod
    def generate_predictors_on_feature(column):
        """
        Given a column, generate all possible threshold for this column
        :param column: a list of values for a feature
        :return: a list of threshold values
        """
        unique_values = np.unique(column)
        unique_values = np.sort(unique_values)
        min_value = unique_values.min()
        max_value = unique_values.max()

        thresholds = [min_value - 0.5]
        if len(unique_values) > 1:
            for i in range(1, len(unique_values)):
                thresholds.append((unique_values[i - 1] + unique_values[i]) / 2.0)
        thresholds.append(max_value + 0.5)
        return thresholds

    def fit(self, train, target, dist):
        """
        Fit the train set to get optimal predictor based on given distribution
        :param train: train data set
        :param target: labels of train data set
        :param dist: weights of each data point
        :return: a predictor that maximum abs(0.5 - error(h))
        """
        max_err = -1.0
        best_error = 0
        best_predictor = None
        best_predicts = None
        for f in self.predictors:
            # print "OptimalWeakLearner: find best predictor on feature: %s with %s thresholds" % (f, len(thresholds))
            for t in self.predictors[f]:
                predictor = FeatureThresholdPredictor(f, t)
                predicts = self.get_predict_result(f, t, predictor, train, target)
                # error_h = self.weighted_error(target, predicts, dist)
                error_h = dist[predicts].sum()
                error = abs(0.5 - error_h)
                if max_err < error:
                    max_err = error
                    best_error = error_h
                    best_predictor = predictor
                    best_predicts = predicts
        return best_predictor, best_error, best_predicts

    @staticmethod
    def weighted_error(target, predicts, dist):
        """
        :param target: actual labels
        :param predicts: predict labels
        :param dist: weights for each data
        """
        err = 0.0
        for p, t, d in zip(target, predicts, dist):
            if p != t:
                err += d
        return err

    def get_predict_result(self, f, t, predictor, train, target):
        """
        Retrieve predict labels from local cache,
        if it is not exists, predict the given data set
        and save the predicted labels into local cache
        :param f: feature index
        :param t: threshold value
        :param predictor: decision stump
        :param train: train data set
        :return a list of predict labels on given data set
        """
        if (f, t) not in self.predict_results:
            self.predict_results[(f, t)] = predictor.predict(train) != target
        return self.predict_results[(f, t)]


class AdaBoost:
    """
    A implementation of AdaBoost algorithm
    """

    def __init__(self):
        pass

    def boost(self, train, train_target, test, test_target, T=100, converged=0.001):
        """
        Running AdaBoost on given dataset
        :param train: train dataset
        :param train_target: train labels
        :param test: test dataset
        :param test_target: test labels
        :param T: maximum of iteration
        :param converged: converge value
        """
        m, n = train.shape
        weights = np.array([1.0 / m] * m)
        weak_leaner = OptimalWeakLearner()
        weak_leaner.setup_predictors(train)

        # final hypothesis
        train_predicts = np.zeros(m)
        test_predicts = np.zeros(len(test))

        for t in range(T):
            predictor, weighted_err, predicts = weak_leaner.fit(train, train_target, weights)
            confidence = 0.5 * np.exp((1.0 - weighted_err) / weighted_err)

            # accumulate final hypothesis
            predicts = predictor.predict(train)
            train_predicts += confidence * predicts
            train_predicts_signed = self.sign(train_predicts)
            er = 1.0
            for actual, pred in zip(train_target, train_predicts_signed):
                if actual != pred:
                    er += 1
            print "iteration %s: feature %s, threshold %s, round_error %s, train_error: %s" % (
                t, predictor.feature, predictor.threshold, weighted_err, er / m)
            # update weights
            for w in range(len(weights)):
                tmp = np.sqrt(weighted_err / (1.0 - weighted_err))
                if train_target[w] != predicts[w]:
                    tmp = np.sqrt((1.0 - weighted_err) / weighted_err)
                weights[w] = (weights[w] * tmp) / (2.0 * np.sqrt(weighted_err * (1 - weighted_err)))

    @staticmethod
    def sign(vals):
        return np.array([-1.0 if v <= 0 else 1.0 for v in vals])