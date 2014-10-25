__author__ = 'jiachiliu'

from boosting.tree import DecisionTree
import random
from validation import confusion_matrix, confusion_matrix_analysis
import numpy as np
from ranking import auc, roc


class Bagging:
    def __init__(self):
        self.round_predicts = []

    def bagging(self, train, train_target, test, test_target, T=50, param=0.05):
        t = 0
        indices = range(len(train))
        choose_size = int(len(train) * param)
        while t < T:
            print "Round %s" % (t + 1)
            choose_indices = random.sample(indices, choose_size)
            X = train[choose_indices]
            Y = train_target[choose_indices]
            dt = DecisionTree()
            dt.fit(X, Y, 4)
            predicts = dt.predict(test)
            self.round_predicts.append(predicts)
            t += 1

        majority_votes = []
        for t in range(len(test)):
            positive = 0
            negative = 0
            for predicts in self.round_predicts:
                if predicts[t] == 1:
                    positive += 1
                else:
                    negative += 1
            if positive > negative:
                majority_votes.append(1.0)
            else:
                majority_votes.append(0.0)

        test_predicts = np.array(majority_votes)
        test_cm = confusion_matrix(test_predicts, test_target)
        test_err, test_acc, test_fpr, test_tpr = confusion_matrix_analysis(test_cm)
        roc_points = roc(test_target, test_predicts)
        test_auc = auc(roc_points[:, 1], roc_points[:, 0])
        print "test error: %s, test acc: %s, auc: %s" % (test_err, test_acc, test_auc)
        return test_acc, test_err, test_auc