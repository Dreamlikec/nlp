# —*- coding: utf-8 -*-

import argparse


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_size", type=int, default=100, help="embedding size of word embedding")
    parser.add_argument("--epoch", type=int, default=5, help="epoch of training")
    parser.add_argument("--cuda", type=bool, default=True, help="whether use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate during training")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size during training")
    parser.add_argument("--min_count", type=int, default=20, help="min count of words to ignore")
    parser.add_argument("--window_size", type=int, default=2, help="window size of sliding")
    parser.add_argument("--x_max", type=int, default=200, help="x_max of glove")
    parser.add_argument("--alpha", type=float, default=0.75, help="alpha index of x_max")

    return parser.parse_args()
