
def get_cocitation_network(dataset, dataset_folder):
    # List to store hyperedges
    hyedges = []

    # Dictionary storing hyperedges
    cocitations = {}

    if dataset == "cora":
        cites_file_handle = open(dataset_folder + "/Cora/cora.cites")
    elif dataset == "citeseer":
        cites_file_handle = open(dataset_folder + "/Citeseer/citeseer.cites")

    for line in cites_file_handle.readlines():
        temp = line.split("\n")
        link = temp[0].split("\t")

        if link[1] not in cocitations:
            cocitations[link[1]] = set([])

        if link[0] != link[1]:
            cocitations[link[1]].add(link[0])

    for key in cocitations:
        if len(cocitations[key]) > 1:
            hyedges.append(list(cocitations[key]))

    return hyedges


def get_coreference_network(dataset, dataset_folder):
    # List to store hyperedges
    hyedges = []

    # Dictionary storing hyperedges
    coreferences = {}

    if dataset == "cora":
        cites_file_handle = open(dataset_folder + "/Cora/cora.cites")
    elif dataset == "citeseer":
        cites_file_handle = open(dataset_folder + "/Citeseer/citeseer.cites")

    for line in cites_file_handle.readlines():
        temp = line.split("\n")
        link = temp[0].split("\t")

        if link[0] not in coreferences:
            coreferences[link[0]] = set([])

        if link[0] != link[1]:
            coreferences[link[0]].add(link[1])

    for key in coreferences:
        if len(coreferences[key]) > 1:
            hyedges.append(list(coreferences[key]))

    return hyedges


def get_cora_citeseer_network(dataset, network, dataset_folder):
    if network == "cocitation":
        hyedges = get_cocitation_network(dataset, dataset_folder)
    elif network == "coreference":
        hyedges = get_coreference_network(dataset, dataset_folder)

    return hyedges

