import argparse


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--embed_size", default=10, type=int)
    parser.add_argument("--epoch", default=300, type=int)
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--learning_rate", default=0.005, type=int)
    parser.add_argument("--max_length", default=100, type=int)
    parser.add_argument("--n_gram", default=2, type=int)
    parser.add_argument("--min_count", default=3, type=int)
    parser.add_argument("--label_num", default=4 ,type=int)

    return parser.parse_args()