__author__ = 'jiachiliu'

from boosting.dataset import load_spambase
from boosting.ActiveLearning import *
from boosting.cross_validation import *
import numpy as np

al = ActiveLearning()
train, target = load_spambase()
target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, target))
train, test, train_target, test_target = train_test_shuffle_split(train, target, len(train) / 10)
al.active_learning(train, train_target, test, test_target)