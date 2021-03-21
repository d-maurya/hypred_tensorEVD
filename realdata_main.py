"""
@author: deepak maurya 
webpage: https://d-maurya.github.io/
code available on: https://github.com/d-maurya/hypred_tensorEVD

Paper: Hyperedge Prediction using Tensor Eigenvalue Decomposition
    https://arxiv.org/pdf/2102.04986.pdf
    
Authors: Deepak Maurya and Balaraman Ravindran
Affiliation: Indian Institute of Technology Madras, India

If you are using this code, please cite using 
@article{maurya2021hyperedge,
  title={Hyperedge Prediction using Tensor Eigenvalue Decomposition},
  author={Maurya, Deepak and Ravindran, Balaraman},
  journal={arXiv preprint arXiv:2102.04986},
  year={2021}
}    

README: 
   run the file 'realdata_main.py' 
   Please change the dataset using the variable 'realdt_argsinp' 
   Please change the folder name using variable 'new_folder' for each run 
   
   the results will be saved in the folder results/<new_folder>/
   it will save multiple files 
       parameters.txt: it will save the parameters like cand_ratio, num_runs, dataset_name etc.. 
       CM_neigh.csv: predictions by common neighbour algorithm
       Katz.csv: predictions by Katz algorithm
       NDP_cand.csv: predictions by HPRA algorithm (NDP := Node Degree Preserved)
       tensor_eig.csv: predictions by tensor eigenvalue decomposition (proposed algorithm in this work https://arxiv.org/pdf/2102.04986.pdf) 
       avg_F1.csv: average F1 scores. A high score indicates the algorithm is performing better.  
"""

import numpy as np
import pandas as pd
import os 
import time 

from realdata_utils import *
from hy_utils import *

from measures import *
from network import get_network
from model_evaluation_1 import *
from all_models import main_model
from gen_candset import *


'''
Dataset options: 
    
'uchoice_bakery', 
'uchoice_Walmart_Depts', 
'contact_highschool' 
'contact_primaryschool', 
'NDC_substances' 
'''

realdt_argsinp = {'dataset_name': 'contact_primaryschool'} 
real_data = get_realdata(realdt_argsinp)
H = real_data['incidence_matrix']

# get hyperedges of specific size only (size 3)
H = get_hy_specific_card(H)

nodes = range(H.shape[0])
m = list(np.unique(np.squeeze(np.asarray(sp.csr_matrix.sum(H, axis=0)))))
argsinp_dict = {'icidence_matrix':H}

orig_hyedges = get_hyperedges_from_incidence(argsinp_dict)
weights_hy = np.ones(len(orig_hyedges))

node_labels = [1]*H.shape[0] # all nodes in one cluster assumed 
hy_labels = [0]*H.shape[1] # as all nodes in one cluster, the hy_labels will be 0 
num_nodes = H.shape[0]
p_del = 0.1 # fraction of hyperedges to be deleted (construction of test set)
cand_ratio = 3 # cand_ratio = (number of non-existing hyperedges in test-set)/(number of existing hyperedges in test-set)
num_runs = 10

new_folder =  'uchoiceWalamrtDepts_iter50_mns8'
os.mkdir('results/'+new_folder) 

method = ["CM_neigh","Katz","NDP_cand"] #,'tensor_eig'
candset_opt = 'mns'

file1 = open('results/'+new_folder+"/parameters.txt","w+") 
file1.write("\n Methods = ")
for m1 in method:
    file1.write(m1+', ')
file1.write('\n Dataset = '+ realdt_argsinp['dataset_name'] + '\n')
file1.write('\n candset_opt = '+ candset_opt + '\n')
file1.write('\n ratio for non existing and existing hyperedges in candidate set = '+ str(cand_ratio) + '\n')
file1.close()

avg_f1_pd = pd.DataFrame(index=np.arange(len(method)), columns = method)

for mc_iter in range(num_runs):
    for algo in method:
        
        # delete a few existing hyperedges from given set of hyperedges 
        del_hy, retained_hy, retained_hy_set = rm_hy_main(nodes,orig_hyedges, weights_hy, hy_labels, p_del)
        
        test_df = pd.DataFrame({'test_hy': del_hy})
        test_df['test_hy'] = test_df['test_hy'].apply(list)
        test_df.to_csv('results/'+new_folder + '/test_hy.csv')
        
        # construct cancdidate set of hyperegdes 
        # candidate set will contain the removed hyperedges from the original set of hyperedges 
        #           and set of non-existing hyperedges. The candidate set will be passed to the trainign model 
        #           without giving the information which hyperegde existed in the training set.
        candset_argsindict = {'nodes':nodes,'retained_hy_set':retained_hy_set,'del_hy':del_hy,
                              'retained_hy':retained_hy, 'weights_hy': weights_hy,'incidence_mat':H, 
                              'cardinality_hy':m,'opt':candset_opt,'true_labels':node_labels, 'cand_ratio': cand_ratio}
        candidate_hy, cand_label = get_candset(candset_argsindict) 
        traininp_dict = {'H':H,'numhy_to_predict': len(del_hy),'train_data':retained_hy,'nodes':nodes, 'AUC':True}
        candset_dict = {'candidate_set':candidate_hy}
        
        # the function 'main_model' will return a predictor function based on the 'algo' 
        # the predictor function 'pred_function' will be later on passed the candidate set and it will predict the hyperegdes 
        pred_function = main_model(traininp_dict,algo)
        predicted_hy, candset_scores = pred_function(candset_dict)
        
        # saving the preidctions 
        pred_df = pd.DataFrame({'pred_hy': predicted_hy})
        pred_df['pred_hy'] = pred_df['pred_hy'].apply(list)
        pred_df.to_csv('results/'+new_folder + '/'+algo+'.csv')
        
        # saving the time needed by each algorithm 
        file1 = open('results/'+new_folder+"/parameters.txt","a") 
        file1.write('\n '+str(algo) + ' done '+ ' at '+ str(time.asctime( time.localtime(time.time()) )))
        file1.close()
           
        # computing the average F1 measure ("measure of goodness")
        avg_f1_, hm_f1_ = compute_f1_main(del_hy, predicted_hy)
            
        avg_f1_pd.loc[mc_iter,algo] = avg_f1_
        
    avg_f1_pd.to_csv('results/'+new_folder + '/avg_F1.csv') # saving the results
        
