# —*- coding: utf-8 -*-

import argparse
import torch


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--device", type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--label_num", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--kernel_sizes", type=list, default=[3, 4, 5])
    parser.add_argument("--out_number", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.004)
    parser.add_argument("--use_pretrained_embed", type=bool, default=True)
    parser.add_argument("--embedding_type", type=str, default="word2vec")
    parser.embeddings_pretrained = None
    return parser.parse_args()
