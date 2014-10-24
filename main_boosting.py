__author__ = 'jiachiliu'

from boosting.dataset import load_spambase
from boosting.cross_validation import train_test_shuffle_split, k_fold_cross_validation, shuffle
from boosting.AdaBoost import *


def optimal_weak_learner():
    print '==============Optimal Weak Learner============'
    train, target = load_spambase()
    train, target = shuffle(train, target)
    target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, target))

    k = 10
    train_size = len(train)
    test_index_generator = k_fold_cross_validation(train_size, k)
    fold = 1
    overall_acc = 0
    overall_error = 0
    overall_auc = 0

    for start, end in test_index_generator:
        print "====================Fold %s============" % fold
        k_fold_train = np.vstack((train[range(0, start)], train[range(end, train_size)]))
        test = train[range(start, end)]
        train_target = np.append(target[range(0, start)], target[range(end, train_size)])
        test_target = target[range(start, end)]

        adaboost = AdaBoost(OptimalWeakLearner())
        acc, err, auc = adaboost.boost(k_fold_train, train_target, test, test_target)

        overall_auc += auc
        overall_acc += acc
        overall_error += err
        fold += 1

    print "Overall test accuracy: %s, overall test error: %s, overall test auc: %s" % (
        overall_acc / k, overall_error / k, overall_auc / k)


def random_weak_learner():
    print '==============Random Weak Learner============'
    train, target = load_spambase()
    train, test, train_target, test_target = train_test_shuffle_split(train, target, len(train) / 10)
    train_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, train_target))
    test_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, test_target))
    adaboost = AdaBoost(RandomChooseLeaner())
    adaboost.boost(train, train_target, test, test_target, T=200)


if __name__ == '__main__':
    random_weak_learner()