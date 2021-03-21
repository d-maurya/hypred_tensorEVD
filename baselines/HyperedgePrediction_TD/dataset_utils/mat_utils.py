import os
import scipy.io as sio
import networkx as nx

from dataset_utils.amazon_network import get_amazon_network
from dataset_utils.aminer_network import get_aminer_network
from dataset_utils.cora_citeseer_network import get_cora_citeseer_network
from dataset_utils.dblp_network import get_dblp_network
from dataset_utils.movielens_network import get_movielens_network
from dataset_utils.twitter_network import get_twitter_network
from hypergraph_utils import get_incidence_matrix


def get_largest_cc(nodes, hyedges):
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for hyedge in hyedges:
        for i in range(len(hyedge)):
            for j in range(i + 1, len(hyedge)):
                G.add_edge(hyedge[i], hyedge[j])

    largest_cc = max(nx.connected_component_subgraphs(G), key=len)

    i = 0
    while i < len(hyedges):
        hyedge = hyedges[i]
        temp = hyedge & largest_cc.nodes()

        if len(temp) > 1:
            hyedges[i] = list(temp)
            i += 1
        else:
            hyedges.remove(hyedge)

    return list(largest_cc.nodes()), hyedges


def store_as_mat(dataset, network):
    dataset_folder = os.getcwd() + "/Raw_Datasets"

    if dataset == "citeseer" or dataset == "cora":
        hyedges = get_cora_citeseer_network(dataset, network, dataset_folder)
    elif dataset == "aminer":
        hyedges = get_aminer_network(network, dataset_folder)
    elif dataset == "dblp":
        hyedges = get_dblp_network(dataset_folder)
    elif dataset == "twitter":
        hyedges = get_twitter_network(1000, dataset_folder)
    elif dataset == "amazon":
        hyedges = get_amazon_network(network, dataset_folder)
    elif dataset == "movielens":
        hyedges = get_movielens_network(dataset_folder)

    nodes = set([])
    for hyedge in hyedges:
        nodes.update(hyedge)

    nodes = list(nodes)
    nodes.sort()

    nodes, hyedges = get_largest_cc(nodes, hyedges)
    nodes.sort()

    S, index_map = get_incidence_matrix(nodes, hyedges)

    output_folder = os.getcwd() + "/Datasets/"
    file_name = dataset
    if network != "":
        file_name += "_" + network

    sio.savemat(output_folder + file_name, {"S": S, "index_map": index_map})
