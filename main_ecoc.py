__author__ = 'jiachiliu'

from boosting.ECOC import ECOC
import numpy as np
from boosting.cross_validation import shuffle


def load_data(path, rows=11314, cols=1754):
    data = np.zeros((rows, cols))
    labels = np.zeros(rows)
    f = open(path)
    lines = f.readlines()
    for r in range(len(lines)):
        words = lines[r].strip()
        if words:
            words = words.split(" ")
            labels[r] = int(words[0])
            for i in range(1, len(words)):
                feature = words[i].split(":")
                col = int(feature[0])
                val = float(feature[1])
                data[r][col] = val
    return data, labels


def print_2d_array(array):
    print '\n'.join(str(r) for r in array)


if __name__ == '__main__':
    ecoc = ECOC(8)
    train, train_target = load_data('data/8newsgroup/train.trec/feature_matrix.txt', rows=11314)
    train, train_target = shuffle(train, train_target)
    print train.shape
    print train_target.shape

    test, test_target = load_data('data/8newsgroup/test.trec/feature_matrix.txt', rows=7532)
    print test.shape
    print test_target.shape

    print_2d_array(ecoc.selected_code.tolist())
    ecoc.train(train, train_target, test, test_target, T=200)
    labels = ecoc.test(test)

    err = 0
    for pred, act in zip(labels, test_target):
        if pred != act:
            err += 1
    print "Total error: %s" % (1.0 * err / len(test_target))