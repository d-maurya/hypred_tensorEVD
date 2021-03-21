import argparse
import time

from model_evaluation_1 import compute_model_f1_score


def model_performance(dataset, network, use_candidateset, K, seed):
    print("Evaluating the model . . .")

    # TODO - compute_model_precision(dataset, network, K, seed)
    if use_candidateset == "false":
        compute_model_f1_score(dataset, network, K, seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["aminer", "citeseer", "cora", "dblp", "twitter", "movielens", "amazon"],
                        help="Dataset to be used - aminer or citeseer or dblp or twitter or movielens or amazon")

    parser.add_argument("--network", type=str, choices=["cocitation", "coreference", "copurchase", "coview", "coauthorship"], default="",
                        help="Network type to be fetched")

    parser.add_argument("--use_candidateset", type=str, choices=["true", "false"],
                        help="whether to use candidate hyedges set or not")

    parser.add_argument("--K", type=int,
                        help="K fold validation")

    parser.add_argument("--seed", type=int, default=0,
                        help="random seed to be used")

    args = parser.parse_args()

    dataset = args.dataset
    network = args.network
    use_candidateset = args.use_candidateset
    K = args.K
    seed = args.seed

    model_performance(dataset, network, use_candidateset, K, seed)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    total_time = end_time - start_time
    print("Time elapsed - " + str(total_time))

    exit()
