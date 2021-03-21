import numpy as np
import scipy.sparse as sp
import pdb

def get_pref_node(H):
    node_degree = 0
    while node_degree == 0:
        d_v = np.squeeze(np.asarray(sp.csr_matrix.sum(H, axis=1)))
        node_id = np.random.choice(np.arange(len(d_v)), replace=True, p=1.0*d_v/np.sum(d_v))

        node_degree = d_v[node_id]

    return node_id


def compute_NHAS(hyedge, hra_scores, nodes_count):
    candidate_nodes = []
    scores = []

    # computing Node-Hyperedge Attachment Scores (NHAS) for all candidate nodes
    for candidate_node in range(nodes_count):
        if len(hyedge & set([candidate_node])) == 0:
            score = 0
            for node in hyedge:
                score += hra_scores[node, candidate_node]
            candidate_nodes.append(candidate_node)
            scores.append(score)

    return candidate_nodes, scores


def predict_hyperedge(H, hra_scores, edge_degree):
    hyedge = set([])
    
    # get node based on Preferential Attachment and add to the hyperedge
    node_id = get_pref_node(H)
    hyedge.add(node_id)

    nodes_count = H.shape[0]

    while len(hyedge) < edge_degree:
        # compute NHAS scores based on HRA scores
        #pdb.set_trace()
        candidate_nodes, scores = compute_NHAS(hyedge, hra_scores, nodes_count)
        scores = [0 if i < 0 else i for i in scores]
        if np.sum(scores) == 0:
            break

        # select next node to be added based on hra scores
        selected_node = np.random.choice(candidate_nodes, replace=True, p=scores/np.sum(scores))
        hyedge.add(selected_node)

    #print(hyedge)
    return hyedge
