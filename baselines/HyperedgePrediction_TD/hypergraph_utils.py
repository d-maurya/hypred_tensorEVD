import numpy as np
import scipy.sparse as sp


def get_nodeid(index_map, index, nodes_count):

    if index not in index_map:
        index_map[index] = nodes_count
        nodes_count += 1

    return index_map[index], nodes_count


def get_incidence_matrix(nodes, hyedges):
    H = sp.lil_matrix((len(nodes), len(hyedges)), dtype=float)

    nodes_seen = 0
    hyedges_seen = 0

    index_map = {}

    for hyedge in hyedges:
        for node in hyedge:
            node_id, nodes_seen = get_nodeid(index_map, node, nodes_seen)
            H[node_id, hyedges_seen] += 1
        hyedges_seen += 1

    return H.tocsr(), index_map


def get_adj_matrix(H):

    # computing inverse hyedge degree Matrix
    d_e = 1.0*np.subtract(np.squeeze(np.asarray(sp.csr_matrix.sum(H, axis=0))), 1)
    D_e_inv = sp.spdiags(np.reciprocal(d_e), [0], H.shape[1], H.shape[1], format="csr")

    # computing node degree preserving reduction's adjacency matrix
    A_ndp = H.dot(D_e_inv.dot(H.transpose()))
    A_ndp = A_ndp - sp.spdiags(A_ndp.diagonal(), [0], A_ndp.shape[0], A_ndp.shape[1], format="csr")

    return A_ndp


def validate_hyedges(hyedges, hyedges_indices, H):
    d_v = np.squeeze(np.asarray(sp.csr_matrix.sum(H.tocsr(), axis=1)))
    valid_nodes = set(np.nonzero(d_v)[0])

    valid_hyedges = []
    valid_hyedges_indices = []

    for i in range(len(hyedges)):
        hyedge = hyedges[i]
        valid = 1
        for node in hyedge:
            if node not in valid_nodes:
                valid = 0
                break

        if valid == 1:
            valid_hyedges.append(hyedge)
            valid_hyedges_indices.append(hyedges_indices[i])

    return valid_hyedges, valid_hyedges_indices


def get_hyedges_from_indices(S, indices):
    S_transpose = S.transpose().tolil()

    hyedges = []
    for index in indices:
        hyedge = S_transpose.rows[index]
        hyedges.append(hyedge)

    return hyedges