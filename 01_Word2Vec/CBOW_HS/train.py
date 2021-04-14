import sys
sys.path.append("../CBOW_HS/")
from data_process import DataProcess
from cbow_hs_model import CBowHSModel
from tqdm import tqdm
from config import ArgumentParser
import torch.optim as optim

config = ArgumentParser()

# hyper parameters
WINDOW_SIZE = config.window_size
BATCH_SIZE = config.batch_size
MIN_COUNT = config.min_count
EMB_DIMENSION = config.embed_dimension
LR = config.learning_rate


class Word2Vec(object):
    def __init__(self, input_file_name, output_file_name):
        self.output_file_name = output_file_name
        self.data = DataProcess(input_file_name, MIN_COUNT)
        self.model = CBowHSModel(self.data.word_count, EMB_DIMENSION)
        self.lr = LR
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self):
        print("CBow Training...")
        pairs_count = self.data.evaluate_pairs_count(WINDOW_SIZE)
        print("pairs_count", pairs_count)
        batch_count = pairs_count / BATCH_SIZE
        print("batch_count", batch_count)
        process_bar = tqdm(range(int(batch_count)))
        for _ in process_bar:
            pos_pairs = self.data.get_batch_pairs(BATCH_SIZE, WINDOW_SIZE)
            pos_pairs, neg_pairs = self.data.get_pairs(pos_pairs)

            pos_u = [pair[0] for pair in pos_pairs]
            pos_w = [pair[1] for pair in pos_pairs]
            neg_u = [pair[0] for pair in neg_pairs]
            neg_w = [pair[1] for pair in neg_pairs]

            self.optimizer.zero_grad()
            loss = self.model.forward(pos_u, pos_w, neg_u, neg_w)
            loss.backward()
            self.optimizer.step()

            process_bar.set_postfix(loss=loss.data)
            process_bar.update()
        self.model.save_embeddings(self.data.id2word_dict, self.output_file_name)


if __name__ == '__main__':
    w2v = Word2Vec(input_file_name='../data/text8.txt', output_file_name="../results/cbow_hs.txt")
    w2v.train()
