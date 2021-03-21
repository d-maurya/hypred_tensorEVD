import scipy.io


def get_amazon_network(network_type, dataset_folder):

    if network_type == "copurchase":
        mat = scipy.io.loadmat(dataset_folder + "/Amazon/copurchase.mat")
        network = mat['copurchase'][0, 0]

    elif network_type == "coview":
        mat = scipy.io.loadmat(dataset_folder + "/Amazon/coview.mat")
        network = mat['coview'][0, 0]

    hyedges = []
    for i in range(len(network)):
        hyedge = network[i][0]
        hyedges.append(list(hyedge))
    
    return hyedges
