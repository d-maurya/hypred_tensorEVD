#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 20:29:46 2020

@author: deepak
"""

import pandas as pd
import scipy.sparse as sp
import numpy as np
import pdb 
import random 

from scipy import optimize
from scipy.special import comb
from itertools import combinations 

from hy_utils import *

from model_evaluation_1 import *


def hypred_prop_teneig(argsinp_dict):
    '''
    function to return the predicted hyperedges - number of hyperedges given 
    - assume to have the candidate set 
    
    calculate the  hyperedge score for candidate hy from the fiedler vector of Laplacian from existing edges 
    
    Please refer https://arxiv.org/pdf/2102.04986.pdf for the algorithm
    '''
    # sparse incidence matrix 
    H = argsinp_dict['H']
    nodes = argsinp_dict['nodes']
    num_hy_pred = argsinp_dict['numhy_to_predict'] # number of hyperedges to be predicted 
    d_e = 1.0*np.squeeze(np.asarray(sp.csr_matrix.sum(H, axis=0)))
    hy_card = d_e[0]
    d_v = 1.0*(np.squeeze(np.asarray(sp.csr_matrix.sum(H, axis=1))))
    
    D_e_inv = sp.spdiags(np.reciprocal(d_e), [0], H.shape[1], H.shape[1], format="csr")
    A_un = H.dot(D_e_inv.dot(H.transpose()))
    A_un = A_un - sp.spdiags(A_un.diagonal(), [0], A_un.shape[0], A_un.shape[1], format="csr") 
    
    hyedges = argsinp_dict['train_data']
    
    def comp_kthord_poly(hy_edges, x_var):
        '''
        compute the Lx^k which is summed over all the hyperedges 

        '''
        df = pd.DataFrame({'hyedge': hy_edges})
        df['x_var'] = df.apply(lambda y: x_var[list(y.hyedge)], axis=1)
        df['score'] = df['x_var'].apply(am_gm_score)
        return df['score'].sum()

        
    def teneig_der_kmin1poly(hy_edges, x_var1, eig_val):
        '''
        compute the derivative: Lx^k-1 - eig_val*x 
        '''
        
        am_deritmp = hy_card*np.power(x_var1, hy_card-1)
        df = pd.DataFrame({'node_num': range(H.shape[0]), 'node_deg': d_v.tolist(),'am_deritmp': am_deritmp})
        df['am_deri'] = df.apply(lambda x: x.node_deg*x.am_deritmp, axis = 1)
        df['hyedge_nodei'] = df.apply(lambda y: H[y.node_num,:].tolil().rows[0], axis = 1)
        
        
        def kminus1_gmfn(row):
            
            df = pd.DataFrame({'hyedge_nodei': row['hyedge_nodei']})
            df['node_num'] = row['node_num']
            H2 = H.transpose() # IMPORTANT not the incidece matrix but its transpose
            df['nodes'] = df.apply(lambda y: H2[y.hyedge_nodei,:].tolil().rows[0], axis = 1)
            
            df['gm_derv'] = df.apply(lambda y: hy_card*np.prod(x_var1[list(y.nodes)])/x_var1[y.node_num], axis = 1)
            return df['gm_derv'].sum()
            
        df['gm_deri'] = df.apply(kminus1_gmfn, axis = 1)
        df['grad'] = df['am_deri'] - df['gm_deri'] - eig_val*x_var1
        return np.linalg.norm(np.asarray(df['grad']))
        
    
    def obj_opti(x):
        return comp_kthord_poly(hyedges, x_var = x[1:])
    
    def constraint_1(x):
        return teneig_der_kmin1poly(hyedges, x[1:], x[0])
        
    def constraint_2(x):
        return -1*comp_kthord_poly(hyedges, x_var = x[1:])
    
    con1 = {'type':'ineq','fun':constraint_1}
    con2 = {'type':'ineq','fun':constraint_2}
    
    cons = (con1,con2)
    
    b_eig = (0,10000)
    b_eigvec = (-1,1)
    bounds_ = flatten((b_eig, (b_eigvec,)*H.shape[0]))
    x0_int = np.random.rand(H.shape[0]+1,)
    result = optimize.minimize(obj_opti,x0=x0_int,bounds = bounds_, method='SLSQP',constraints=cons,options={'maxiter':50})
    fiedler_vec = result.x
    
    def teneig_score(row):
        '''
        score of one hyperedge
        '''
        hyperedge = row['cand_set'] 
        e_vec = fiedler_vec
    
        score_hy = pd.DataFrame()
        score_hy['edges'] =  [hyperedge]
        score_hy['edges'] =  score_hy['edges'].apply(list)
        score_hy['fiedl_vec'] = score_hy.apply(lambda x: e_vec[x.edges], axis=1) 
        score_hy['score'] = score_hy['fiedl_vec'].apply(am_gm_score)
        score = score_hy['score'].sum() # NO NEED
        return score
    
    def pred_func(candset_dict):   
        '''
        a function to return the most probable occuring hyperedges basec on the scores
        '''
        candset_list = candset_dict['candidate_set']
        hy_score = pd.DataFrame()
        hy_score['cand_set'] = candset_list
        
        hy_score['score'] = hy_score.apply(teneig_score, axis = 1)
        all_scores = hy_score['score'].values.tolist()
        hy_score = hy_score.sort_values('score')
        
        hy_score = hy_score.reset_index(drop=True)
        threshold_score = hy_score['score'][num_hy_pred-1]
        pred_hy = hy_score.loc[hy_score['score'] <= threshold_score]
        
        pred_hy['cand_set_set'] = pred_hy['cand_set'].apply(set)
        predicted_hyperedges = list(pred_hy['cand_set_set'])
        return predicted_hyperedges, all_scores
    return pred_func  


def common_neigh(argsinp_dict):
    
    H = argsinp_dict['H']
    num_hy_pred = argsinp_dict['numhy_to_predict'] 
    A_un = H.dot(H.transpose())
    
    def CM_neigh(row):
        hyperedge = row['cand_set']
        hy_red = list(combinations(hyperedge, 2)) # reduced hyperedge 
        score_hy = pd.DataFrame()
        score_hy['edges'] =  hy_red
        score_hy['score'] =  score_hy.apply(lambda x: (A_un[x.edges[0]].multiply(A_un[x.edges[1]])).sum(), axis=1) 
        score = 1.0*score_hy['score'].sum()/len(hyperedge)
        return score 
    
    def pred_func(candset_dict):
        
        candset_list = candset_dict['candidate_set']
        hy_score = pd.DataFrame()
        hy_score['cand_set'] = candset_list
        hy_score['score'] = hy_score.apply(CM_neigh, axis = 1)
        all_scores = hy_score['score'].values.tolist()
        
        hy_score = hy_score.sort_values('score')
        
        hy_score = hy_score.reset_index(drop=True)
        threshold_score = hy_score['score'][num_hy_pred-1]
        pred_hy = hy_score.loc[hy_score['score'] <= threshold_score]
        
        pred_hy['cand_set_set'] = pred_hy['cand_set'].apply(set)
        predicted_hyperedges = list(pred_hy['cand_set_set'])
        return predicted_hyperedges, all_scores
    
    return pred_func

def Katz(argsinp_dict):
    '''
    computing the {paths of length exactly l from x to y}
    '''
    H = argsinp_dict['H']
    num_hy_pred = argsinp_dict['numhy_to_predict'] 
    A_un = H.dot(H.transpose())
    num_nodes = A_un.shape[0]
    beta_param = 0.0001
    dist_sim = sp.linalg.inv(sp.identity(num_nodes, format="csr") - beta_param*A_un) - sp.identity(num_nodes, format="csr") 
    def hy_katz(row):
        hyperedge = row['cand_set']
        hy_red = list(combinations(hyperedge, 2)) # reduced hyperedge 
        score_hy = pd.DataFrame()
        score_hy['edges'] =  hy_red
        score_hy['score'] =  score_hy.apply(lambda x: (dist_sim[x.edges[0]].multiply(dist_sim[x.edges[1]])).sum(), axis=1) 
        score = 1.0*score_hy['score'].sum()/len(hyperedge)
        return score 
    
    def pred_func(candset_dict):
        
        candset_list = candset_dict['candidate_set']
        hy_score = pd.DataFrame()
        hy_score['cand_set'] = candset_list
        hy_score['score'] = hy_score.apply(hy_katz, axis = 1)
        all_scores = hy_score['score'].values.tolist()
        
        hy_score = hy_score.sort_values('score')
        
        hy_score = hy_score.reset_index(drop=True)
        threshold_score = hy_score['score'][num_hy_pred-1]
        pred_hy = hy_score.loc[hy_score['score'] <= threshold_score]
        
        pred_hy['cand_set_set'] = pred_hy['cand_set'].apply(set)
        predicted_hyperedges = list(pred_hy['cand_set_set'])
        return predicted_hyperedges, all_scores
    
    return pred_func

def hypred_ndp_cand(argsinp_dict):
    # self written candidate version 
    
    H = argsinp_dict['H']
    retained_hy = argsinp_dict['train_data']
    num_hy_pred = argsinp_dict['numhy_to_predict'] 
    hra_scores = get_hra_scores(H)
    hyedges_degree, hyedges_degree_frequencies = get_hyedges_degree_dist(H)
    
    def hy_ndp(row):
        hyperedge = row['cand_set']
        hy_red = list(combinations(hyperedge, 2)) # reduced hyperedge 
        score_hy = pd.DataFrame()
        score_hy['edges'] =  hy_red
        score_hy['score'] =  score_hy.apply(lambda x: (hra_scores[x.edges[0]].multiply(hra_scores[x.edges[1]])).sum(), axis=1) 
        score = 1.0*score_hy['score'].sum()/len(hyperedge)
        return score 

    def pred_func(candset_dict):
        
        candset_list = candset_dict['candidate_set']
        hy_score = pd.DataFrame()
        hy_score['cand_set'] = candset_list
        hy_score['score'] = hy_score.apply(hy_ndp, axis = 1)
        all_scores = hy_score['score'].values.tolist()
        
        hy_score = hy_score.sort_values('score')
        
        hy_score = hy_score.reset_index(drop=True)
        threshold_score = hy_score['score'][num_hy_pred-1]
        pred_hy = hy_score.loc[hy_score['score'] <= threshold_score]
        
        pred_hy['cand_set_set'] = pred_hy['cand_set'].apply(set)
        predicted_hyperedges = list(pred_hy['cand_set_set'])
        return predicted_hyperedges, all_scores
    
    
    return pred_func
   
    
def main_model(argsinp_dict, algo):
    '''
    return a predictor function and then 
    '''
    if algo == 'tensor_eig':
        pred_function = hypred_prop_teneig(argsinp_dict)
    elif algo == "NDP_cand":
        pred_function = hypred_ndp_cand(argsinp_dict)
    elif algo == "CM_neigh":
        pred_function = common_neigh(argsinp_dict)
    elif algo == "Katz":
        pred_function = Katz(argsinp_dict)
    else:
        raise NotImplementedError
    return pred_function
