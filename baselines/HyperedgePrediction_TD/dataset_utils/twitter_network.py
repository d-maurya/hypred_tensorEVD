def get_twitter_network(no_nodes, dataset_folder):
    higgsfile = open(dataset_folder + '/higgs-social_network.edgelist', 'r')

    hypergraph = {}
    for line in higgsfile.readlines():
        temp = line.split("\n")
        link = temp[0].split(" ")

        head = int(link[0])
        tail = int(link[1])

        if tail not in hypergraph.keys():
            hypergraph[tail] = []

        hypergraph[tail].append(head)

    keys = list(hypergraph.keys())
    keys.sort()

    trunc_nodes = keys[0:no_nodes]

    hyedges = []
    for node in trunc_nodes:
        hyedge = hypergraph[node]
        new_hyedge = []
        for i in range(len(hyedge)):
            if hyedge[i] in trunc_nodes:
                new_hyedge.append(hyedge[i])

        if len(new_hyedge) > 1:
            hyedges.append(new_hyedge)

    return hyedges