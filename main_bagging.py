__author__ = 'jiachiliu'

from boosting.dataset import load_spambase
from boosting.cross_validation import *
from boosting.Bagging import Bagging

train, target = load_spambase()
train, test, train_target, test_target = train_test_shuffle_split(train, target, len(train) / 10)
res = []
param = 1
print "Choose %.2f%% of data" % (param * 100)
bagging = Bagging()
acc, err, auc = bagging.bagging(train, train_target, test, test_target, T=50, param=param)
res.append((acc, err, auc))
param += 0.05
print res

