from torch.utils import data
import os
import random
import numpy as np

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors


class MR_Dataset(data.Dataset):
    def __init__(self, state="train", k=0, embedding_type="word2vec"):
        self.path = os.path.abspath(".")
        if "data" not in self.path:
            self.path += "/data"
        # 导入数据
        pos_samples = open(self.path + "/MR/rt-polarity.pos", errors="ignore").readlines()
        neg_samples = open(self.path + "/MR/rt-polarity.neg", errors="ignore").readlines()
        datas = pos_samples + neg_samples
        datas = [data.split() for data in datas]
        max_sample_length = max([len(sample) for sample in datas])
        labels = [1] * len(pos_samples) + [0] * len(neg_samples)
        word2id = {"<pad>": 0}
        for i, data in enumerate(datas):
            for j, word in enumerate(data):
                if word2id.get(word) is None:
                    word2id[word] = len(word2id)
                datas[i][j] = word2id[word]
            datas[i] = datas[i] + [0] * (max_sample_length - len(datas[i]))
        self.vocab_size = len(word2id)
        self.word2id = word2id
        self.embeddings_weights = None

        if embedding_type == "word2vec":
            self.get_word2vec()
        elif embedding_type == "glove":
            self.get_glove()
        else:
            pass
        data_bag = list(zip(datas, labels))
        random.seed(1)
        random.shuffle(data_bag)
        datas, labels = zip(*data_bag)

        if state == "train":
            self.datas = datas[:int(k * len(datas) / 10)] + datas[int((k + 1) * len(datas) / 10):]
            self.labels = labels[:int(k * len(labels) / 10)] + labels[int((k + 1) * len(labels) / 10):]
            self.datas = np.array(self.datas[0:int(len(self.datas) * 0.9)])
            self.labels = np.array(self.labels[0:int(len(self.labels) * 0.9)])
        elif state == "valid":
            self.datas = datas[:int(k * len(datas) / 10)] + datas[int((k + 1) * len(datas) / 10):]
            self.labels = labels[:int(k * len(labels) / 10)] + labels[int((k + 1) * len(labels) / 10):]
            self.datas = np.array(self.datas[int(len(self.datas) * 0.9):])
            self.labels = np.array(self.labels[int(len(self.labels) * 0.9):])
        elif state == "test":
            self.datas = np.array(datas[int(k * len(datas) / 10): int((k + 1) * len(datas) / 10)])
            self.labels = np.array(labels[int(k * len(labels) / 10): int((k + 1) * len(labels) / 10)])

    def __getitem__(self, item):
        return self.datas[item], self.labels[item]

    def __len__(self):
        return len(self.datas)

    def get_word2vec(self):
        """
        生成word2vec词向量，这里使用的是skip—gram
        """
        if not os.path.exists(self.path + "/word2vec_embeddings_mr.npy"):
            print("Reading word2vec Embeddings...")
            w2vmodel = KeyedVectors.load_word2vec_format(self.path + "/GoogleNews-vectors-negative300.bin.gz",
                                                         binary=True)
            tmp = []
            for word, index in self.word2id.items():
                try:
                    tmp.append(w2vmodel.get_vector(word))
                except:
                    pass
            tmp = np.array(tmp)
            mean = np.mean(tmp)
            std = np.std(tmp)
            embeddings_weights = np.random.normal(mean, std, [self.vocab_size, 300])
            for word, index in self.word2id.items():
                try:
                    embeddings_weights[index, :] = w2vmodel.get_vector(word)
                except:
                    pass
            np.save(self.path + "/word2vec_embeddings_mr.npy", embeddings_weights)
        else:
            embeddings_weights = np.load(self.path + "/word2vec_embeddings_mr.npy")
        self.embeddings_weights = embeddings_weights

    def get_glove(self):
        if not os.path.exists(self.path + "/glove_embeddings_mr.npy"):
            if not os.path.exists(self.path + "./glove_word2vec.txt"):
                glove_file = datapath(self.path + "/glove.840B.300d.txt")
                tmp_file = get_tmpfile(self.path + "/glove_word2vec.txt")
                from gensim.scripts.glove2word2vec import glove2word2vec
                glove2word2vec(glove_file, tmp_file)
            else:
                tmp_file = get_tmpfile(self.path + "/glove_word2vec.txt")

            print(tmp_file)

            print("Reading Glove Embeddings...")

            w2vmodel = KeyedVectors.load_word2vec_format(tmp_file)
            tmp = []
            for word, index in self.word2id.items():
                try:
                    tmp.append(w2vmodel.get_vector(word))
                except:
                    pass
            tmp = np.array(tmp)
            mean = np.mean(tmp)
            std = np.std(tmp)
            embeddings_weights = np.random.normal(mean, std, [self.vocab_size, 300])
            for word, index in self.word2id.items():
                try:
                    embeddings_weights[index, :] = w2vmodel.get_vector(word)
                except:
                    pass
            np.save(self.path + "/glove_embeddings_mr.npy")
        else:
            embeddings_weights = np.load(self.path + "/glove_embeddings_mr.npy")

        self.embeddings_weights = embeddings_weights


if __name__ == "__main__":
    mr_train_dataset = MR_Dataset()
    print(len(mr_train_dataset))
    mr_valid_dataset = MR_Dataset("valid")
    print(len(mr_valid_dataset))
    mr_test_dataset = MR_Dataset("test")
    print(len(mr_test_dataset))
