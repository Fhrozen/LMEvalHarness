
from collections import defaultdict
from sklearn import metrics


def balanced_mean(arr):
    # each entry is of the form (acc score, class label)
    # first group the results
    by_class = defaultdict(list)
    for acc, label in arr:
        by_class[label].append(acc)

    # calculate class averages
    avgs = []
    for key, vals in by_class.items():
        avgs.append(sum(vals) / len(vals))

    # average the class values
    return sum(avgs) / len(avgs)


def macro_f1(items):
    # this is different from f1-score which uses default binary avg
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = metrics.f1_score(golds, preds, average="macro")

    return fscore
