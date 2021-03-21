import random
import numpy as np
import scipy.sparse as sp
import pdb 

from collections import Counter
from hyperedge_prediction import predict_hyperedge
from hypergraph_utils import get_adj_matrix, get_incidence_matrix, validate_hyedges, get_hyedges_from_indices
from measures import compute_avg_f1_score
from network import get_network


def get_missing_hyedges_indices(hyedges_count, K):
    missing_hyedges_indices_list = []
    missing_hyedges_count = hyedges_count / float(K)

    shuffled_indices = list(range(0, hyedges_count))
    random.shuffle(shuffled_indices)
    for k in range(K):
        missing_hyedges_indices_list.append(shuffled_indices[round(missing_hyedges_count * k):round(missing_hyedges_count * (k + 1))])

    return missing_hyedges_indices_list


def get_hra_scores(H):
    
    d_v = np.squeeze(np.asarray(sp.csr_matrix.sum(H, axis=1)))
    d_v_inv = 1. / d_v
    d_v_inv[d_v_inv == np.inf] = 0
    D_v_inv = sp.diags(d_v_inv)
    # pdb.set_trace()
    A_ndp = get_adj_matrix(H)

    print("computing HRA scores")
    hra_scores = A_ndp + (A_ndp.dot(D_v_inv)).dot(A_ndp)

    return hra_scores


def get_hyedges_degree_dist(H):
    hyedges_degree_dist = Counter(np.squeeze(np.asarray(sp.csr_matrix.sum(H, axis=0))))
    hypedges_count = H.shape[1]

    hyedges_degree = []
    hyedges_degree_frequencies = []
    for hyedge_degree in hyedges_degree_dist.keys():
        hyedges_degree.append(int(hyedge_degree))
        hyedges_degree_frequencies.append(hyedges_degree_dist[hyedge_degree] / hypedges_count)

    return hyedges_degree, hyedges_degree_frequencies


def compute_model_f1_score(dataset, network, K, seed):
    random.seed(seed)

    # get the incidence matrix
    S = get_network(dataset, network)
    hyedges_count = S.shape[1]

    # get missing hyedges list used for K-fold cross validation
    missing_hyedges_indices_list = get_missing_hyedges_indices(hyedges_count, K)

    f1_score_list = []
    for k in range(K):
        print("round - " + str(k))

        missing_hyedges_indices = missing_hyedges_indices_list[k]
        existing_hyedges_indices = np.sort(list(set(list(range(S.shape[1]))) - set(missing_hyedges_indices)))

        print("getting incidence matrix")
        H = S[:, existing_hyedges_indices]

        # checking whether the missing hyedges contain any singleton nodes
        print("validating missing hyedges")
        missing_hyedges = get_hyedges_from_indices(S, missing_hyedges_indices)
        valid_missing_hyedges, valid_missing_hyedges_indices = validate_hyedges(missing_hyedges, missing_hyedges_indices, H)

        print("getting pairwise scores")
        hra_scores = get_hra_scores(H)

        print("getting the hyedges degree distribution")
        hyedges_degree, hyedges_degree_frequencies = get_hyedges_degree_dist(H)

        print("predicting missing hyedges")
        predicted_hyedges = []
        for j in range(len(valid_missing_hyedges)):
            sample_hyedge_degree = np.random.choice(hyedges_degree, replace=True, p=hyedges_degree_frequencies)
            predicted_hyedges.append(predict_hyperedge(H, hra_scores, sample_hyedge_degree))

        print("computing f1 score")
        f1_score = compute_avg_f1_score(valid_missing_hyedges, predicted_hyedges)

        f1_score_list.append(f1_score)
        print("f1 score - " + str(f1_score))

    print("\n")
    print("EVALUATION DETAILS ")

    print("HPRA MODEL RESULTS ")
    print("f1 score list - ")
    print(f1_score_list)

    print("average f1 score - " + str(np.sum(f1_score_list)/len(f1_score_list)))
    print("std - f1 score - " + str(np.std(f1_score_list, dtype=np.float64)))

