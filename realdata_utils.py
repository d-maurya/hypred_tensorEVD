"""
Created on Tue Jun 30 03:26:58 2020

@author: deepak
"""
import scipy.io 
import pandas as pd 
import matplotlib.pyplot as plt
import networkx as nx 

from hy_utils import *

def get_largest_cc(nodes, hyedges):
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for hyedge in hyedges:
        for i in range(len(hyedge)):
            for j in range(i + 1, len(hyedge)):
                G.add_edge(hyedge[i], hyedge[j])
    
    #comp_len = [len(c) for c in sorted(nx.connected_component_subgraphs(G),key=len, reverse=True)]
    # largest_cc = max(nx.connected_component_subgraphs(G), key=len)
    cc = [G.subgraph(c) for c in nx.connected_components(G)]
    cc_len = [len(G_sub) for G_sub in cc]
    max_pos = cc_len.index(max(cc_len))
    largest_cc = cc[max_pos]
    
    
    i = 0
    while i < len(hyedges):
        hyedge = hyedges[i]
        temp = hyedge & largest_cc.nodes()
        
        if (i % 100 == 0):
            print(i)
            
        if len(temp) > 1:
            hyedges[i] = list(temp)
            i += 1
        else:
            hyedges.remove(hyedge)
    
    return list(largest_cc.nodes()), hyedges

def get_hy_specific_card(H):
    df = pd.DataFrame({'hy_num': range(H.shape[1])})
    df['hyedge'] = df.apply(lambda row: list(H[:,row.hy_num].nonzero()[0]), axis=1) 
    df['hy_card'] = df['hyedge'].apply(len)
    df = df.loc[df['hy_card'] == 3] #
    
    hyedges = df['hyedge'].values.tolist()
    nodes = set([])
    for hyedge in hyedges:
        nodes.update(hyedge)
    nodes = list(nodes)
    nodes, hyedges = get_largest_cc(nodes, hyedges)
    
    weight = [1]*len(hyedges)
    H, index_map = get_incidence_matrix_w(nodes, hyedges, weight)
        
    return H
    
def get_hyperedges_from_incidence(argsinp_dict):
    H = argsinp_dict['icidence_matrix'] # csr format
    df = pd.DataFrame({'hy_num': range(H.shape[1])})
    df['hyedge'] = df.apply(lambda row: list(H[:,row.hy_num].nonzero()[0]), axis=1) 
    
    return df['hyedge'].values.tolist()
    
    
def get_amazon_network(network_type):

    if network_type == "copurchase":
        mat = scipy.io.loadmat('Datasets/'+ 'Amazon/copurchase.mat')
        network = mat['copurchase'][0, 0]

    elif network_type == "coview":
        mat = scipy.io.loadmat('Datasets/'+ 'Amazon/coview.mat')
        network = mat['coview'][0, 0]

    hyedges = []
    for i in range(len(network)):
        hyedge = network[i][0]
        hyedges.append(list(hyedge))
    
    return hyedges


def get_realdata(realdt_argsinp):
    '''
    main function to load the preprocessed real data
    '''
    
    dataset_name = realdt_argsinp['dataset_name']
    realdata_argsout = {}
    
    if (dataset_name == 'uchoice_bakery'):
        H = scipy.io.loadmat('Datasets/'+ 'uchoice_bakery.mat')['S']
    elif (dataset_name == 'uchoice_Walmart_Depts'):
        H = scipy.io.loadmat('Datasets/'+ 'uchoice_Walmart_Depts.mat')['S']
    elif (dataset_name ==  'contact_highschool'):
        H = scipy.io.loadmat('Datasets/'+ 'contact_highschool.mat')['S']
    elif (dataset_name ==  'contact_primaryschool'):
        H = scipy.io.loadmat('Datasets/'+ 'contact_primaryschool.mat')['S']
    elif (dataset_name ==  'NDC_substances'):
        H = scipy.io.loadmat('Datasets/'+ 'NDC_substances.mat')['S']
        
    H = H.tocsr()
    realdata_argsout['incidence_matrix'] = H
    return realdata_argsout
    
def plot_hyedgefreq(file_location, dataset_name, orig_hyedges):
    df = pd.DataFrame({'hy':orig_hyedges})
    df['hy_size'] = df['hy'].apply(len)
    df['hy_freq'] = df.groupby(['hy_size'])['hy'].transform('count')
    
    plt.figure(0)
    plt.bar(x = df['hy_size'], height = df['hy_freq'], color = 'midnightblue')
    plt.title('Hyperedge Cardinality Frequency for ' + dataset_name + ' dataset')
    plt.savefig(file_location + '/' +dataset_name + '.png')
    return 0    