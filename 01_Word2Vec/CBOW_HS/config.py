import argparse


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=4, help="window_size in word2vec")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size during training model")
    parser.add_argument("--min_count", type=int, default=3, help="min count of training word")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--embed_dimension", type=int, default=100, help="embedding dimension of word embedding")
    return parser.parse_args()
