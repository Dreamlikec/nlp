import sys
sys.path.append("../CBOW_NEG")

from cbow_neg_model import CBowNEGModel
from tqdm import tqdm
from config import ArgumentParser
from data_process import DataProcess
import torch.optim as optim

config = ArgumentParser()
BATCH_SIZE = config.batch_size
WINDOW_SIZE = config.window_size
LEARNING_RATE = config.learning_rate
EMB_DIMENSION = config.embed_dimension
MIN_COUNT = config.min_count
NEG_COUNT = config.neg_count


class Word2Vec(object):
    def __init__(self, input_file_name, output_file_name):
        self.output_file_name = output_file_name
        self.data = DataProcess(input_file_name, MIN_COUNT)
        self.model = CBowNEGModel(self.data.word_count, EMB_DIMENSION)
        self.lr = LEARNING_RATE
        self.optimizer = optim.SGD(self.model.parameters(), self.lr)

    def train(self):
        print("SkipGram Training...")
        pairs_count = self.data.evaluate_pairs_count(WINDOW_SIZE)
        print("pairs_count", pairs_count)
        batch_count = pairs_count / BATCH_SIZE
        print("batch_count", batch_count)
        process_bar = tqdm(range(int(batch_count)))

        for _ in process_bar:
            pos_pair = self.data.get_batch_pairs(BATCH_SIZE, WINDOW_SIZE)
            pos_w = [pair[0] for pair in pos_pair]
            pos_v = [int(pair[1]) for pair in pos_pair]
            neg_v = self.data.get_negative_sampling(pos_pair, NEG_COUNT)

            self.optimizer.zero_grad()
            loss = self.model.forward(pos_w, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()
            process_bar.set_postfix(loss=loss.data)
            process_bar.update()
        self.model.save_embeddings(self.data.id2word_dict, self.output_file_name)


if __name__ == '__main__':
    w2v = Word2Vec(input_file_name='../data/text8.txt', output_file_name="skip_gram_hs.txt")
    w2v.train()
