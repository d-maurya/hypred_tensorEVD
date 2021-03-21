#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:03:49 2020

@author: deepak
"""
import pandas as pd
import random 
import scipy.sparse as sp
import numpy as np
import pdb 
from itertools import combinations 
import sys
import networkx as nx 
from sklearn import metrics

sys.path.insert(1,'baselines/HyperedgePrediction_TD/')

#from hypergraph_utils import get_incidence_matrix
from measures import compute_f1_score


def flatten(test_tuple): 
      
    if isinstance(test_tuple, tuple) and len(test_tuple) == 2 and not isinstance(test_tuple[0], tuple): 
        res = [test_tuple] 
        return tuple(res) 
  
    res = [] 
    for sub in test_tuple: 
        res += flatten(sub) 
    return tuple(res) 

def compute_auc(hyperedges_class, hyperedges_score):
    fpr, tpr, thresholds = metrics.roc_curve(hyperedges_class, hyperedges_score)
    auc = metrics.auc(fpr, tpr)

    return auc

def am_gm_score(x):
    '''
    compute the am gm score for uniform hypergraphs
    input x is a list 
    '''
    m = len(x)
    score = abs(np.sum(np.power(x,m)) - m*np.prod(x))
    return score

def get_incidence_from_hy(nodes, hyperedges):
    
    H = sp.lil_matrix((len(nodes), len(hyperedges)), dtype=float)    
    
    for i,hy in enumerate(hyperedges):
        H[hy,i] = 1
    
    return H.tocsr()


def get_nodeid(index_map, index, nodes_count):

    if index not in index_map:
        index_map[index] = nodes_count
        nodes_count += 1

    return index_map[index], nodes_count

def harmonic_mean(vec_list):
    a = np.array(vec_list)
    return (len(a) / np.sum(1.0/a))

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

def get_incidence_matrix_w(nodes, hyedges, weights):
    H = sp.lil_matrix((len(nodes), len(hyedges)), dtype=float)

    nodes_seen = 0
    hyedges_seen = 0

    index_map = {}

    for i,hyedge in enumerate(hyedges):
        for node in hyedge:
            node_id, nodes_seen = get_nodeid(index_map, node, nodes_seen)
            H[node_id, hyedges_seen] = weights[i]
        hyedges_seen += 1

    return H.tocsr(), index_map

def rm_hy(nodes, orig_hyedges, weights_hy, hy_labels, p_del):
    '''
    Function: to delete few existing hyperedges 
    Works for non-uniform hypergraphs and hence we can use it for uniform hypergraphs also 
    
    inputs: 
    nodes - a list of nodes in the graph 
    orig_hyedges - a list of hyperedges in a hypergaph in the beginning 
    weights_hy - the weights of those hyperedges
    hy_labels - 0 if they belog to intra class, non-zero if they are interclass
    p_del - fraction or precentage of hyperedges to be deleted 
    
    outputs: 
        deleted_hy
        retained_hy 
        retained_hy_set
    '''
    hy_ = pd.DataFrame()
    hy_['all_hy'] = orig_hyedges
    
    # pdb.set_trace()
    H = get_incidence_from_hy(nodes, orig_hyedges)
    d_v = np.squeeze(np.asarray(sp.csr_matrix.sum(H, axis=1)))
    
    hy_['min_deg'] = hy_.apply(lambda row: min(d_v[list(row.all_hy)]), axis=1)
    hy_['labels'] = hy_labels
    mean_degree = np.mean(d_v)
    const = mean_degree #np.std(d_v) #
    
    hy = hy_.loc[hy_['min_deg'] >= (mean_degree-const)]
    hy = hy.loc[hy['labels'] == 0] # filtering only intra community hyperedges 
    num_del_hy = min(np.int(p_del*len(orig_hyedges)), hy.shape[0]) 

    del_hyindx = random.sample(list(hy.index.values), num_del_hy)
    del_hy = hy.loc[del_hyindx]
    del_hy['all_hy_set'] = del_hy['all_hy'].apply(set)
    deleted_hy = list(del_hy['all_hy_set'])
    
    hy_['all_hy_hashed'] = hy_['all_hy'].map(lambda x: hash(frozenset(x)))
    del_hy['all_hy_hashed'] = del_hy['all_hy'].map(lambda x: hash(frozenset(x)))
    
    all_hyperedges = pd.merge(hy_, del_hy, on = 'all_hy_hashed', how = 'left')
    ret_hy = all_hyperedges[all_hyperedges['labels_y'].isnull()]
    ret_hy['all_hy_set'] = ret_hy['all_hy_x'].apply(set)
    retained_hy = ret_hy['all_hy_x'].values.tolist()
    retained_hy_set = ret_hy['all_hy_set'].values.tolist()
    return deleted_hy, retained_hy, retained_hy_set

def rm_hy_main(nodes, orig_hyedges, weights_hy, hy_labels, p_del):
    '''
    this is supposed to be for non-uniform hypergraphs 
    where we pass each uniform hypergraph to the rm_hy function 
    '''
    deleted_hy= retained_hy= retained_hy_set = []
    df_hy = pd.DataFrame({'hy_edges': orig_hyedges, 'weights': weights_hy, 'hy_label': hy_labels})
    df_hy['hy_size'] = df_hy['hy_edges'].apply(len)
    df_hy_grp = df_hy.groupby('hy_size')
    m = df_hy_grp.groups.keys()
    
    for m_i in m:
        orig_hyedges_mi = df_hy_grp.get_group(m_i)['hy_edges'].values.tolist()
        weights_hy_mi = df_hy_grp.get_group(m_i)['weights'].values.tolist()
        hy_labels_mi = df_hy_grp.get_group(m_i)['hy_label'].values.tolist()
        deleted_hy_mi, retained_hy_mi, retained_hy_set_mi = rm_hy(nodes, orig_hyedges_mi, weights_hy_mi, hy_labels_mi, p_del)
        deleted_hy = deleted_hy_mi + deleted_hy
        retained_hy =  retained_hy_mi + retained_hy
        retained_hy_set += retained_hy_set_mi
    
    return deleted_hy, retained_hy, retained_hy_set


def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 
    
def f1_score_self(del_hy, predicted_hyedges):
    #result_pred = sorted(set(map(tuple, predicted_hyedges)), reverse=True)
    #result_rm = sorted(set(map(tuple, del_hy)), reverse=True)
    #common_part = set(result_rm) & set(result_pred)
    common_part = intersection(del_hy, predicted_hyedges)
    f1_score = 0

    if len(common_part) > 0:
        true_positives = len(common_part)
        total_positives = len(predicted_hyedges)
        actual_positives = len(del_hy)
        
        precision = 1.0*true_positives / total_positives
        recall = 1.0*true_positives / actual_positives

        f1_score = (2 * precision * recall) / (precision + recall)
        
    return f1_score

def compute_hm_f1_score(hyperedges, predicted_hyperedges):
    '''
    1st arguement - missing hyperedges (truth)
    2nd - predicted hyperedges 
    '''
    avg_f1_score_1 = 0.0
    for predicted_hyperedge in predicted_hyperedges:
        max_f1_score = 0.0
        for hyperedge in hyperedges:
            f1_score = compute_f1_score(hyperedge, predicted_hyperedge)
            if f1_score > max_f1_score:
                max_f1_score = f1_score

        avg_f1_score_1 += max_f1_score
    
    avg_f1_score_1 = 0 if len(predicted_hyperedges) == 0  else avg_f1_score_1 / len(predicted_hyperedges)

    avg_f1_score_2 = 0
    for hyperedge in hyperedges:
        max_f1_score = 0

        for predicted_hyperedge in predicted_hyperedges:
            f1_score = compute_f1_score(hyperedge, predicted_hyperedge)
            if f1_score > max_f1_score:
                max_f1_score = f1_score

        avg_f1_score_2 += max_f1_score

    avg_f1_score_2 = avg_f1_score_2 / len(hyperedges)
    
    avg_f1_score_2 = 0 if len(hyperedges) == 0  else avg_f1_score_2 / len(hyperedges)

    avg_f1_score = (avg_f1_score_1 + avg_f1_score_2)/2
    
    if avg_f1_score_1 == 0:  avg_f1_score_1 = 1e-4 
    if avg_f1_score_2 == 0:  avg_f1_score_2 = 1e-4 
    
    hm_f1_score = 2*(avg_f1_score_1 * avg_f1_score_2)/(avg_f1_score_1 + avg_f1_score_2)

    return avg_f1_score, hm_f1_score

def compute_f1_main(hyperedges, predicted_hyperedges):
    '''
    Parameters
    ----------
    hyperedges : a list of sets which were removed from original set of hyperedges to generate 
                    test set
    predicted_hyperedges : a list of sets containing predicted hyperedges by any algorithm

    Returns
    -------
    the average F1 metric for computed using the harmonic mean of all the average F1 for each 
        cardinality of hyperedge present in test set 

    Working: filter all the hyperedges of cardinality k (present in test set), from both test set 
              and predicted set
              further compute the average F1 for each cardinality k 
              compute the harmonic mean of average F1s  
    '''
    df_test = pd.DataFrame({'hy': hyperedges})
    df_pred = pd.DataFrame({'hy': predicted_hyperedges})
    
    df_test['hy_size'] = df_test['hy'].apply(len)
    df_pred['hy_size'] = df_pred['hy'].apply(len)
    
    df_test_grp = df_test.groupby('hy_size')
    m = df_test_grp.groups.keys()
    avg_f1 = hm_f1 = []
     
    for m_i in m:
        test_hy = df_test_grp.get_group(m_i)['hy'].values.tolist()
        pred_hy = (df_pred['hy'].loc[df_pred['hy_size'] == m_i]).values.tolist()
        avg_f1_m_i, hm_f1_m_i = compute_hm_f1_score(test_hy, pred_hy)
        avg_f1 = avg_f1 + [avg_f1_m_i]
        hm_f1 = hm_f1 + [hm_f1_m_i]
    
    avg_f1 = [1e-4 if x==0 else x for x in avg_f1]
    hm_f1 = [1e-4 if x==0 else x for x in hm_f1]
    # compute harmonic mean 
    avg_f1_score = harmonic_mean(avg_f1)    
    hm_f1_score = harmonic_mean(hm_f1)    
    return avg_f1_score, hm_f1_score

    
def hy_reduction(argsinp_dict):
    H = argsinp_dict['incidence_mat']
    # W = argsinp_dict['weights_hy']
    A_un = H.dot(H.transpose())
    
    A_un = A_un - sp.spdiags(A_un.diagonal(), [0], A_un.shape[0], A_un.shape[1], format="csr")
    
    #Gr = nx.from_numpy_matrix(A_un.toarray())
    Gr = nx.from_scipy_sparse_matrix(A_un)
    return Gr

    
def hedges_to_pg(hedges, weights, p):
    #print("Generating {}-pg..".format(p))

    assert (p >= 2)

    pg = nx.Graph()

    node_size = p - 1

    n_edges = 0
    for idx, hedge in enumerate(hedges):

        if len(hedge) < p:
            continue

        w = weights[idx]

        if p == 2:
            nodes = hedge
        else:
            nodes = []
            for node in combinations(hedge, node_size):
                nodes.append(node)

        # Update edges
        for edge in combinations(nodes, 2):
            node1, node2 = edge

            if p > 2:
                if len(set(node1 + node2)) > p:
                    continue

            if pg.has_edge(node1, node2):
                pg[node1][node2]['weight'] += w
            else:
                pg.add_edge(node1, node2, weight=w)
                n_edges += 1
                if n_edges % (500 * 1000) == 0:
                    print(n_edges)
                if n_edges == 500 * 1000 * 1000:  # cannot handle more
                    print("Cannot handle more")
                    return pg

    # print("..done")
    return pg    

    