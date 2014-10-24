__author__ = 'jiachiliu'


from boosting.dataset import load_crx
from boosting.cross_validation import train_test_shuffle_split
from boosting.AdaBoost import *

data, target = load_crx()
train, test, train_target, test_target = train_test_shuffle_split(data, target, len(data) / 10)
train_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, train_target))
test_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, test_target))

adaboost = AdaBoost(OptimalWeakLearner())
adaboost.boost(train, train_target, test, test_target, discrete_features=[0,3,4,5,6,8,9,11,12])