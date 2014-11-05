__author__ = 'jiachiliu'

from boosting.dataset import load_spambase
from boosting.cross_validation import train_test_shuffle_split
from boosting.AdaBoost import *

train, target = load_spambase()
target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, target))
train, test, train_target, test_target = train_test_shuffle_split(train, target, len(train) / 10)

percentage = 0.05
increment = 0.05
increment_size = int(len(train) * increment)
init_size = int(len(train) * 0.05)
indices = range(len(train))

init_dataset = train[:init_size]
init_target = train_target[:init_size]

remain_dataset = train[init_size:]
remain_target = train_target[init_size:]


# Active learning
X = init_dataset
Y = init_target
R = remain_dataset
RY = remain_target
result = []
while percentage < 0.5:
    print "labeled data: %.2f%%" % (100.0 * len(X) / len(train))
    adaboost = AdaBoost(OptimalWeakLearner())
    acc, err, auc = adaboost.boost(X, Y, test, test_target)
    result.append((acc, err, auc))
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
    percentage += increment



# boosting
init_dataset = train[:init_size]
init_target = train_target[:init_size]
print "init_dataset length: %s" % len(init_dataset)

percentage = 0.05
res = []
X = init_dataset
Y = init_target
indices = range(len(train))
while percentage < 0.5:
    print "Choose %.2f%% of data" % (percentage * 100)
    adaboost = AdaBoost(OptimalWeakLearner())
    acc, err, auc = adaboost.boost(X, Y, test, test_target)
    res.append((acc, err, auc))
    percentage += 0.05
    choose_size = int(len(indices) * percentage)
    choose_indices = random.sample(indices, choose_size)

    X = train[choose_indices]
    Y = train_target[choose_indices]

# boosting - random
init_dataset = train[:init_size]
init_target = train_target[:init_size]
print "init_dataset length: %s" % len(init_dataset)

percentage = 0.05
res_random = []
X = init_dataset
Y = init_target
indices = range(len(train))
while percentage < 0.5:
    print "Choose %.2f%% of data" % (percentage * 100)
    adaboost = AdaBoost(RandomChooseLeaner())
    acc, err, auc = adaboost.boost(X, Y, test, test_target)
    res_random.append((acc, err, auc))
    percentage += 0.05
    choose_size = int(len(indices) * percentage)
    choose_indices = random.sample(indices, choose_size)

    X = train[choose_indices]
    Y = train_target[choose_indices]

print result
print res
print res_random