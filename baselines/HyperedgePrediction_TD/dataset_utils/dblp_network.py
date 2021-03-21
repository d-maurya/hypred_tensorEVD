import pickle


def get_dblp_network(dataset_folder):
    dblp_file = open(dataset_folder + '/DBLP/dblp.pickle', 'rb')
    dblp_dataset = pickle.load(dblp_file)

    intralayers = dblp_dataset['intra']

    hyedges = []
    for i in range(len(intralayers)):
        hyedge_info = intralayers[i]

        head_set = set(hyedge_info[0])
        tail_set = set(hyedge_info[1])

        hyedge = list(head_set | tail_set)
        hyedge.sort()
        hyedges.append(hyedge)

    return hyedges


