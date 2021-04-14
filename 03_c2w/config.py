# —*- coding: utf-8 -*-
import argparse


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_chars", type=int, default=512)
    parser.add_argument("--char_embed_size", type=int, default=50)
    parser.add_argument("--max_word_length", type=int, default=16)
    parser.add_argument("--max_sentence_length", type=int, default=100)
    parser.add_argument("--char_hidden_size", type=int, default=150)
    parser.add_argument("--lm_hidden_size", type=int, default=150)
    parser.add_argument("--word_embed_size", type=int, default=50)
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--learning_rate", type=int, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=100)
    return parser.parse_args()