def get_file_paths(file, dataset_folder):
    file_handle = open(file, 'r')

    file_list = []
    for file in file_handle.readlines():
        file_list.append(dataset_folder + file[:-1])

    return file_list


def get_aminer_coreference(dataset_folder):

    # Set to store all referenced paper ids
    reference_ids = set([])

    # Dictionary to store coreferences of a paper (dictionary storing hyedges)
    coreferences = {}

    coreferences_list = []

    # Dictionary to store paper id to class id mapping
    paperid_classid = {}

    curr_classid = 1

    file_list = get_file_paths("Raw_Datasets/ArnetMiner/filePaths.txt", dataset_folder)
    for file in file_list:

        file_handle = open(file, encoding="ISO-8859-1")
        for line in file_handle.readlines():

            line_type = line[1]

            if line_type == 'i':
                index = line[6:-1]
                paperid_classid[index] = curr_classid

            elif line_type == '%':
                if line[2] != ' ':
                    ref_index = line[2:-1]
                    reference_ids.add(ref_index)

                    if ref_index not in coreferences:
                        coreferences[ref_index] = []
                    coreferences[ref_index].append(index)

        curr_classid += 1

    for key in coreferences:
        if len(coreferences[key]) != 0:
            coreferences_list.append(list(coreferences[key]))

    return paperid_classid, reference_ids, coreferences_list


def get_aminer_cocitation(dataset_folder):

    # Set to store all referenced paper ids
    reference_ids = set([])

    # List to store referenced paper ids (dictionary storing hyedges)
    references_list = []

    # Dictionary to store paper id to class id mapping
    paperid_classid = {}

    curr_classid = 1

    file_list = get_file_paths("Raw_Datasets/ArnetMiner/filePaths.txt", dataset_folder)
    for file in file_list:

        file_handle = open(file, encoding="ISO-8859-1")
        for line in file_handle.readlines():

            line_type = line[1]

            if line_type == 'i':
                index = line[6:-1]
                paperid_classid[index] = curr_classid
                references = set([])

            elif line_type == '%':
                if line[2] != ' ':
                    ref_index = line[2:-1]
                    references.add(ref_index)

            elif line_type == '!':
                if len(references) > 1:
                    references_list.append(list(references))
                    reference_ids.update(references)

        curr_classid += 1

    return paperid_classid, reference_ids, references_list


def prune_hyedges(labeled_nodes, referenced_nodes, hyedges):

    common_nodes = labeled_nodes & referenced_nodes

    i = 0
    while i < len(hyedges):
        hyedge = hyedges[i]
        temp = set(hyedge) & common_nodes

        if len(temp) > 1:
            hyedges[i] = list(temp)
            i += 1
        else:
            hyedges.remove(hyedge)

    return hyedges


def get_aminer_network(network, dataset_folder):
    
    if network == "cocitation":
        paperid_classid, reference_ids, references_list = get_aminer_cocitation(dataset_folder)
    elif network == "coreference":
        paperid_classid, reference_ids, references_list = get_aminer_coreference(dataset_folder)
    
    pruned_hyedges = prune_hyedges(paperid_classid.keys(), reference_ids, references_list)

    return pruned_hyedges