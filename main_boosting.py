__author__ = 'jiachiliu'

from boosting.dataset import load_spambase
from boosting.cross_validation import train_test_shuffle_split, k_fold_cross_validation, shuffle
from boosting.AdaBoost import *
import pylab as plt


def optimal_weak_learner_on_random_data():
    data, target = load_spambase()
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
        if fold == 10:
            hypo = adaboost.hypothesis(test)
            roc_points = roc(test_target, hypo, 1.0, -1.0)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.xlim(xmin=0)
            plt.ylim(ymin=0)
            plt.scatter(roc_points[:, 1], roc_points[:, 0])
            plt.show()

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
    optimal_weak_learner()