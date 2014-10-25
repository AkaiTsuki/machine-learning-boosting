__author__ = 'jiachiliu'

from boosting.dataset import load_crx
from boosting.cross_validation import train_test_shuffle_split
from boosting.AdaBoost import *

data, target = load_crx()
train, test, train_target, test_target = train_test_shuffle_split(data, target, len(data) / 10)
train_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, train_target))
test_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, test_target))

indices = range(len(train))
param = 0.05
res = []
while param < 0.5:
    print "Choose %.2f%% of data" % (param * 100)
    choose_size = int(len(indices) * param)
    choose_indices = random.sample(indices, choose_size)

    X = train[choose_indices]
    Y = train_target[choose_indices]

    adaboost = AdaBoost(OptimalWeakLearner())
    acc, err, auc = adaboost.boost(X, Y, test, test_target, discrete_features=[0, 3, 4, 5, 6, 8, 9, 11, 12])
    res.append((acc, err, auc))
    param += 0.05

print res

