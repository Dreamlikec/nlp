import argparse


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--embed_dimension", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--min_count", type=int, default=3)
    return parser.parse_args()
