from __future__ import division
from sklearn import metrics
import pdb 

def compute_auc(hyperedges_class, hyperedges_score):
    fpr, tpr, thresholds = metrics.roc_curve(hyperedges_class, hyperedges_score)
    auc = metrics.auc(fpr, tpr)

    return auc


def compute_precision(hyperedges_class, hyperedges_size):
    precision = 0.0

    for i in range(hyperedges_size):
        precision += hyperedges_class[len(hyperedges_class) - 1 - i]
    precision = (precision/hyperedges_size)

    return precision


def compute_f1_score(hyperedge, predicted_hyperedge):
    #pdb.set_trace()
    common_part = set(hyperedge) & set(predicted_hyperedge)

    f1_score = 0

    if len(common_part) > 0:
        true_positives = len(common_part)
        total_positives = len(predicted_hyperedge)
        actual_positives = len(hyperedge)

        precision = true_positives / total_positives
        recall = true_positives / actual_positives

        f1_score = (2 * precision * recall) / (precision + recall)

    return f1_score


def compute_avg_f1_score(hyperedges, predicted_hyperedges):
    # pdb.set_trace()
    avg_f1_score_1 = 0.0
    for predicted_hyperedge in predicted_hyperedges:
        max_f1_score = 0.0
        for hyperedge in hyperedges:
            f1_score = compute_f1_score(hyperedge, predicted_hyperedge)
            if f1_score > max_f1_score:
                max_f1_score = f1_score

        avg_f1_score_1 += max_f1_score

    avg_f1_score_1 = avg_f1_score_1 / len(predicted_hyperedges)

    avg_f1_score_2 = 0
    for hyperedge in hyperedges:
        max_f1_score = 0

        for predicted_hyperedge in predicted_hyperedges:
            f1_score = compute_f1_score(hyperedge, predicted_hyperedge)
            if f1_score > max_f1_score:
                max_f1_score = f1_score

        avg_f1_score_2 += max_f1_score

    avg_f1_score_2 = avg_f1_score_2 / len(hyperedges)

    avg_f1_score = (avg_f1_score_1 + avg_f1_score_2)/2

    return avg_f1_score
