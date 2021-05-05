import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score


def f1_score(num_classes, predicted, labels):
    """
    Считает метрику F1 по предсказанным значениям и истинным меткам класса
    input: predicted - предсказанные значения, labels - истинные значения
    output: число типа float, соответствующее метрике F1
    """
    data = (dict(Counter(sorted(zip(labels, predicted)))))
    conf = np.zeros((num_classes, num_classes))  # conf - confusion matrix
    for i in range(num_classes):
        for j in range(num_classes):
            if (i, j) in data.keys():
                conf[i, j] = data[(i, j)]
    print(conf)

    f1 = np.zeros(shape=(num_classes,), dtype='float32')
    weights = np.sum(conf, axis=0) / np.sum(conf)
    for ind_class in range(num_classes):
        tp = conf[ind_class, ind_class]
        fp = np.sum(conf[ind_class, np.concatenate((np.arange(0, ind_class), np.arange(ind_class + 1, num_classes)))])
        fn = np.sum(conf[np.concatenate((np.arange(0, ind_class), np.arange(ind_class + 1, num_classes))), ind_class])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[ind_class] = 2 * precision * recall / (precision + recall + 1e-7) * weights[ind_class] if (precision + recall + 1e-7) > 0 else 0
    f1 = np.sum(f1)
    return f1

pred = np.random.rand(32).round()
print(pred)
targets = np.random.randint(2, size=32)
print(targets)
#print(f1_score(1, pred, targets))
print(accuracy_score(targets, pred))