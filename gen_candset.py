#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 00:28:59 2020

@author: deepak

This code is for generating the negative samples for hyperedge prediction 
"""
from itertools import combinations 
import random 
import pandas as pd
import numpy as np
import scipy.sparse as sp


from hy_utils import am_gm_score,get_incidence_from_hy, hedges_to_pg, hy_reduction

def sample_neqcliques_m(size, G, exclude, n, max_iter):
    """
    For m-uniform hypergraph 
    
    df_: dataframe 
    G: reduced graph in nx format
    
    
    :param size: clique size
    :param G: nx graph
    :param exclude: exclude these
    :param n: try to find n cliques
    :param strong: strong clique
    :param max_iter: try until this iteration of while loop
    :return:
    """
    
    cliques = set()

    n_neighbors = size - 1
    n_edges = size * (size - 1) / 2

    nodes = list(G.nodes)

    n_iter = 0
    while len(cliques) < n:
        if len(cliques) >= n:
            break

        if n_iter >= max_iter:
            break
        #if n_iter % (10 * 1000) == 0:
            #print("n_iter: {}".format(n_iter))

        n_iter += 1

        node = random.choice(nodes)  # select a random node

        all_neighbors = [neigh for neigh in G.neighbors(node)]
        if len(all_neighbors) < n_neighbors:
            continue

        # find at most 1 clique
        for i in range(10):

            neighbors = random.sample(all_neighbors, k=n_neighbors)

            subn = [node] + neighbors
            subg = G.subgraph(subn)

            if subg.number_of_edges() != n_edges:
                continue  # not a clique

            clique = subn

            clique.sort()
            clique = tuple(clique)

            if clique not in exclude:
                cliques.add(clique)
                break

    cliques = list(cliques)  # list of non-overlapping cliques

    #pdb.set_trace()
    #df_c = pd.DataFrame({'neg_hy': cliques})
    #df_c['hy_size'] = size 
    return cliques


def sample_neqcliques_m_degree(size, G, exclude, n, max_iter):
    """
    For m-uniform hypergraph for degree distribution 
    
    df_: dataframe 
    G: reduced graph in nx format
    
    
    :param size: clique size
    :param G: nx graph
    :param exclude: exclude these
    :param n: try to find n cliques
    :param strong: strong clique
    :param max_iter: try until this iteration of while loop
    :return:
    """
    
    cliques = set()

    n_neighbors = size - 1
    n_edges = size * (size - 1) / 2

    nodes = list(G.nodes)
    df = pd.DataFrame({'retained_hyset':exclude})
    df['hyedges'] = df['retained_hyset'].apply(list)
    hyperedges = df['hyedges'].values.tolist()
    H = get_incidence_from_hy(nodes, hyperedges)
    d_v = 1.0*(np.squeeze(np.asarray(sp.csr_matrix.sum(H, axis=1))))
    degree_probab = d_v/sum(d_v)
    
    n_iter = 0
    while len(cliques) < n:
        if len(cliques) >= n:
            break

        if n_iter >= max_iter:
            break
        #if n_iter % (10 * 1000) == 0:
            #print("n_iter: {}".format(n_iter))

        n_iter += 1

        # node = random.choice(nodes)  # select a random node
        node = np.random.choice(nodes, replace=True, p=degree_probab)
        all_neighbors = [neigh for neigh in G.neighbors(node)]
        if len(all_neighbors) < n_neighbors:
            continue

        # find at most 1 clique
        for i in range(10):

            neighbors = random.sample(all_neighbors, k=n_neighbors)

            subn = [node] + neighbors
            subg = G.subgraph(subn)

            if subg.number_of_edges() != n_edges:
                continue  # not a clique

            clique = subn

            clique.sort()
            clique = tuple(clique)

            if clique not in exclude:
                cliques.add(clique)
                break

    cliques = list(cliques)  # list of non-overlapping cliques

    #pdb.set_trace()
    #df_c = pd.DataFrame({'neg_hy': cliques})
    #df_c['hy_size'] = size 
    return cliques    

def sample_mns(size, G, n_neg):
    '''
    ref: https://link.springer.com/chapter/10.1007/978-3-030-47436-2_46
    MNS sampling algorithm 
    Parameters
    ----------
    size : hyperedge cardinality
    G : reduced hyperrgaph 
    n_neg : number of negative hyperedges 

    Returns
    -------
    a list of n hyperedges of given size

    '''
    if size == 2:
        nodes = G.nodes
        def rand_sample(m, nodes): return (random.sample(nodes, m))
        all_hy = pd.DataFrame()
        all_hy['m'] =  [size]*int(n_neg)
        all_hy['cand_hy'] = all_hy['m'].apply(rand_sample, nodes = nodes)
        hyedges = all_hy['cand_hy'].values.tolist()
        return hyedges
    
    hyedges = []
    edges_G = list(G.edges)
    for i in range(n_neg):
        e_0 = edges_G[np.random.choice(range(len(edges_G)), replace=True)]
        f = set(e_0)
        while len(f) != size:
            neigh_edges_f = list(G.edges(f))
            # have to check the size of the 
            e = neigh_edges_f[np.random.choice(range(len(neigh_edges_f)), replace=True)]  # random choice 
            f = f.union(e) # might be the existing edge
        hyedges = hyedges + [list(f)]
    
    return hyedges
            
            

def get_candset_muni(argsinp_dict):
    '''
    works for uniform hypergraph with cardinality m 
    this function is called multiple times for non-uniform hypergraphs
    get candidate set using all hyperegdes - hyperedges in the training data
    the labels of the training set and candidate set are passed for the computation of AUC metric 
    '''
    nodes = argsinp_dict['nodes']
    retained_hy_set = argsinp_dict['retained_hy_set']
    del_hy = argsinp_dict['del_hy'] # test set 
    m = argsinp_dict['cardinality_hy'] 
    cand_ratio = argsinp_dict['cand_ratio']  # ratio for non-existing hyperedges by existing hyperedges 
    if (argsinp_dict['opt'] == 'all'):
        # get candidate set 
        all_hy = pd.DataFrame()
        all_hy['cand_hy'] = list(combinations(nodes, m))
        all_hy['cand_hy'] = all_hy['cand_hy'].apply(set)
        
        # ref : https://stackoverflow.com/questions/58293776/merge-two-dataframes-based-on-column-containing-set
        all_hy['cand_hy_hashed'] = all_hy['cand_hy'].map(lambda x: hash(frozenset(x)))
        retain_hy = pd.DataFrame({'cand_hy':retained_hy_set})
        retain_hy['flag'] = 1
        retain_hy['cand_hy_hashed'] = retain_hy['cand_hy'].map(lambda x: hash(frozenset(x)))
        cand_hy = pd.merge(all_hy, retain_hy, on = 'cand_hy_hashed', how = 'left')
        cand_hy = cand_hy[cand_hy['flag'].isnull()]
        del_hy_pd = pd.DataFrame({'cand_hy':del_hy})
        del_hy_pd['flag_del'] = 1
        del_hy_pd['cand_hy_hashed'] = del_hy_pd['cand_hy'].map(lambda x: hash(frozenset(x)))
        cand_hy = pd.merge(cand_hy, del_hy_pd, on ='cand_hy_hashed', how = 'left')
        cand_hy['flag_del'] = cand_hy['flag_del'].fillna(0)
        candidate_hy = cand_hy['cand_hy_x'].values.tolist()
        cand_label = cand_hy['flag_del'].values.tolist()
    
    elif (argsinp_dict['opt'] == 'mns'):
        hy_m = retained_hy_set
        G = argsinp_dict['reduced_graph']
        neg_m = sample_mns(m, G, len(hy_m)*cand_ratio)
        all_hy = pd.DataFrame({'cand_hy': neg_m})
        all_hy['cand_hy'] = all_hy['cand_hy'].apply(set)
        # ref : https://stackoverflow.com/questions/58293776/merge-two-dataframes-based-on-column-containing-set
        # removing if there are same hyperedges in the training set and test set 
        all_hy['cand_hy_hashed'] = all_hy['cand_hy'].map(lambda x: hash(frozenset(x)))
        retain_hy = pd.DataFrame({'cand_hy':retained_hy_set})
        retain_hy['flag'] = 1
        retain_hy['cand_hy_hashed'] = retain_hy['cand_hy'].map(lambda x: hash(frozenset(x)))
        cand_hy = pd.merge(all_hy, retain_hy, on = 'cand_hy_hashed', how = 'left')
        cand_hy = cand_hy[cand_hy['flag'].isnull()]
        del_hy_pd = pd.DataFrame({'cand_hy':del_hy})
        del_hy_pd['flag_del'] = 1
        del_hy_pd['cand_hy_hashed'] = del_hy_pd['cand_hy'].map(lambda x: hash(frozenset(x)))
        cand_hy = pd.merge(cand_hy, del_hy_pd, on ='cand_hy_hashed', how = 'left')
        cand_hy['flag_del'] = cand_hy['flag_del'].fillna(0)
        candidate_hy = cand_hy['cand_hy_x'].values.tolist()
        cand_label = cand_hy['flag_del'].values.tolist()
        hy_m = retained_hy_set
        G = argsinp_dict['reduced_graph']
        neg_m = sample_neqcliques_m_degree(m, G, hy_m , len(hy_m)*cand_ratio, max_iter = len(hy_m)*cand_ratio*2)
        cand_hy = pd.DataFrame({'cand_hy': neg_m})
        cand_hy['cand_hy_hashed'] = cand_hy['cand_hy'].map(lambda x: hash(frozenset(x)))
        
        del_hy_pd = pd.DataFrame({'cand_hy':del_hy})
        del_hy_pd['flag_del'] = 1
        del_hy_pd['cand_hy_hashed'] = del_hy_pd['cand_hy'].map(lambda x: hash(frozenset(x)))
        cand_hy = pd.merge(cand_hy, del_hy_pd, on ='cand_hy_hashed', how = 'left')
        cand_hy['flag_del'] = cand_hy['flag_del'].fillna(0)
        candidate_hy = cand_hy['cand_hy_x'].values.tolist()
        cand_label = cand_hy['flag_del'].values.tolist()
        
    return candidate_hy, cand_label

def get_candset(argsinp_dict):
    
    # extracting variables from the input dictionary 
    candidate_hy= cand_label=[];
    retained_hy_set = argsinp_dict['retained_hy_set'] # all hyperedges 
    #retained_hy  = argsinp_dict['retained_hy']
    #weights_hy = argsinp_dict['weights_hy']
    H = argsinp_dict['incidence_mat']
    
    # can filter test set with one cardinality and supply it accordingly 
    del_hy = argsinp_dict['del_hy']
    argsinp_dict_muni = argsinp_dict
    delhy_pd = pd.DataFrame({'del_hy': del_hy})
    delhy_pd['hy_size'] = delhy_pd['del_hy'].apply(len)
    delhy_pd_grp = delhy_pd.groupby('hy_size')
    m_test = delhy_pd_grp.groups.keys()
    
    df_hy = pd.DataFrame({'hy_edges_set': retained_hy_set})
    df_hy['hy_size'] = df_hy['hy_edges_set'].apply(len)
    df_hy_grp = df_hy.groupby('hy_size')
    #m = df_hy_grp.groups.keys()
    
    hyred_args = {'incidence_mat':H}
    argsinp_dict_muni['reduced_graph'] = hy_reduction(hyred_args)
    
    for m_i in m_test:
        argsinp_dict_muni['cardinality_hy'] = m_i
        argsinp_dict_muni['del_hy'] = delhy_pd_grp.get_group(m_i)['del_hy'].values.tolist()
        argsinp_dict_muni['retained_hy_set'] = df_hy_grp.get_group(m_i)['hy_edges_set'].values.tolist()
        candidate_hy_t, cand_label_t = get_candset_muni(argsinp_dict_muni)
        candidate_hy = candidate_hy_t + candidate_hy
        cand_label =  cand_label_t + cand_label
        
    return candidate_hy, cand_label

def sample_neqcliques(G, hyperedges):
    '''
    G is the reduced graph 
    hyperedges is a list of all the cardinalities
    
    output: negative hyperedges not existing in the hypergraph. 
    a list of hyperedges (hyperedges are not set)
    '''
    df_hy = pd.DataFrame({'hy_edges': hyperedges})
    df_hy['hy_size'] = df_hy['hy_edges'].apply(len)
    df_hy_grp = df_hy.groupby('hy_size')
    m = df_hy_grp.groups.keys()
    neq_hy = []
    
    for m_i in m:
        hy_i = df_hy_grp.get_group(m_i)['hy_edges']
        tmp_neg = sample_neqcliques_m(m_i, G, hy_i , len(hy_i)*10, max_iter = len(hy_i)*20)
        neq_hy += tmp_neg
        
    return neq_hy