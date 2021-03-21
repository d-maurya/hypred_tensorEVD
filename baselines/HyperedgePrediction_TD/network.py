import os
import scipy.io


def get_network(dataset, network):
    dataset_folder = os.getcwd() + "/dataset_utils/Datasets/"
    file_path = dataset_folder + dataset + "_" + network + ".mat"
    mat = scipy.io.loadmat(file_path)

    S = mat['S']

    return S