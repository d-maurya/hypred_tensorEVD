# hypred_tensorEVD
This repo contains the code for hyperedge prediction using tensor eigenvalue decomposition. We reperesent the hypergraph as tensor and use the Fiedler vector from Laplacian tensor to predict the new hyperedges using the hyperedge score proposed in our work. 

We compare the results on 5 datasets across three baselines: common neighbour, Katz, and <a href="https://arxiv.org/pdf/2006.11070.pdf">[HPRA]</a> (a resource allocation based algorithm). The code of HPRA algorithm is directly taken from <a href="https://github.com/darwk/HyperedgePrediction ">[this link]</a>.    

For more info about the proposed algorithm, please refer <a href="https://arxiv.org/pdf/2102.04986.pdf">[Hyperedge Prediction using Tensor Eigenvalue Decomposition]</a>. 

# Running the code
  - Run the file 'realdata_main.py' <br>
  - Please change the dataset using the variable 'realdt_argsinp' <br>
  - Please change the folder name using variable 'new_folder' for each run <br>
   
   The results will be saved in the folder results/<new_folder>/. It will save multiple files 
       - parameters.txt: it will save the parameters like cand_ratio, num_runs, dataset_name etc.. <br>
       - CM_neigh.csv: predictions by common neighbour algorithm <br>
       - Katz.csv: predictions by Katz algorithm <br>
       - NDP_cand.csv: predictions by HPRA algorithm (NDP := Node Degree Preserved) <br>
       - tensor_eig.csv: predictions by tensor eigenvalue decomposition (proposed algorithm in this work https://arxiv.org/pdf/2102.04986.pdf) <br>
       - avg_F1.csv: average F1 scores. A high score indicates the algorithm is performing better. <br>


