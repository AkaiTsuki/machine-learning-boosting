__author__ = 'jiachiliu'

import random
from boosting.AdaBoost import *


class ActiveLearning:
    def __init__(self):
        # save the index of data point that already selected
        self.selected = {}
        self.result = []

    def active_learning(self, train, train_target, test, test_target):
        param = 0.05
        increment = 0.05
        init_size = int(len(train) * param)
        increment_size = int(len(train) * increment)

        X = train[:init_size]
        Y = train_target[:init_size]
        R = train[init_size:]
        RY = train_target[init_size:]

        while param < 0.5:
            print "labeled data: %.2f%%" % (100.0 * len(X)/len(train))
            adaboost = AdaBoost(OptimalWeakLearner())
            acc, err, auc = adaboost.boost(X, Y, test, test_target)
            self.result.append((acc, err, auc))
            H = adaboost.hypothesis(R)
            H_abs = np.abs(H)
            sorted_indices = H_abs.argsort().tolist()
            selected = sorted_indices[:increment_size]
            remained = sorted_indices[increment_size:]

            X = np.vstack((X, R[selected]))
            # Y = np.append(Y, adaboost.sign(H[selected]))
            Y = np.append(Y, RY[selected])
            R = R[remained]
            RY = RY[remained]
            param += increment


