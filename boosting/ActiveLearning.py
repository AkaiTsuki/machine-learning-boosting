__author__ = 'jiachiliu'

import random
from boosting.AdaBoost import *


class ActiveLearning:
    def __init__(self):
        # save the index of data point that already selected
        self.selected = {}

    def active_learning(self, train, train_target, test, test_target):
        init_size = int(len(train) * 0.05)
        increment_size = int(len(train) * 0.02)
        upper_size = int(len(train) * 0.5)

        X = train[:init_size]
        Y = train_target[:init_size]
        R = train[init_size:]
        RY = train_target[init_size:]

        while len(X) < upper_size:
            print "Current train set size: %s " % len(X)
            adaboost = AdaBoost(OptimalWeakLearner())
            adaboost.boost(X, Y, test, test_target)
            H = adaboost.hypothesis(R)
            H_abs = np.abs(H)
            sorted_indices = H_abs.argsort().tolist()
            selected = sorted_indices[: increment_size]
            remained = sorted_indices[increment_size:]

            X = np.vstack((X, R[selected]))
            # Y = np.append(Y, adaboost.sign(H[selected]))
            Y = np.append(Y, RY[selected])
            R = R[remained]
            RY = RY[remained]


