__author__ = 'jiachiliu'

from boosting.dataset import load_spambase
from boosting.cross_validation import train_test_shuffle_split
from boosting.AdaBoost import *

print '==============Optimal Weak Learner============'
train, target = load_spambase()
train, test, train_target, test_target = train_test_shuffle_split(train, target, len(train) / 10)
train_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, train_target))
test_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, test_target))
adaboost = AdaBoost(OptimalWeakLearner())
adaboost.boost(train, train_target, test, test_target)

print '==============Random Weak Learner============'

train, target = load_spambase()
train, test, train_target, test_target = train_test_shuffle_split(train, target, len(train) / 10)
train_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, train_target))
test_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, test_target))
adaboost = AdaBoost(RandomChooseLeaner())
adaboost.boost(train, train_target, test, test_target, T=200)