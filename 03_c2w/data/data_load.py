from torch.utils.data import Dataset
import os
import numpy as np
import pickle
from collections import Counter


class CharLMDataSet(Dataset):
    def __init__(self, mode="train", max_word_length=16, max_sentence_length=100):
        self.path = os.path.abspath(".")
        if "data" not in self.path:
            self.path += "/data"
        self.mode = mode
        self.max_word_length = max_word_length
        self.max_sentence_length = max_sentence_length
        self.word2id = {}
        self.char2id = {}
        datas = self.read_files()
        words_datas, char_datas, weights = self.generate_data_label(datas)
        self.words_datas = words_datas.reshape([-1])
        self.char_datas = char_datas.reshape([-1, self.max_word_length])
        self.weights = weights
        print(self.words_datas.shape, self.char_datas.shape, self.weights.shape)

    def __getitem__(self, item):
        return self.char_datas[item], self.words_datas[item], self.weights[item]

    def __len__(self):
        return len(self.words_datas)

    def read_files(self):
        if self.mode == "train":
            datas = open(self.path + "/train.txt", encoding="utf-8").read().strip("\n").splitlines()
            datas = [s.split() for s in datas]
            if not os.path.exists(self.path + "/word2id"):
                words = []
                chars = []
                for data in datas:
                    for word in data:
                        words.append(word.lower())
                        chars.extend(word)
                words = dict(Counter(words).most_common(5000 - 2))
                chars = dict(Counter(chars).most_common(512 - 3))

                word2id = {"<pad>": 0, "<unk>": 1}
                for word in words:
                    word2id[word] = len(word2id)
                char2id = {"<pad>": 0, "<unk>": 1, "<start>": 2}
                for char in chars:
                    char2id[char] = len(char2id)

                self.word2id = word2id
                self.char2id = char2id
                pickle.dump(self.word2id, open(self.path + "/word2id", "wb"))
                pickle.dump(self.char2id, open(self.path + "/char2id", "wb"))
            else:
                self.word2id = pickle.load(open(self.path + "/word2id", "rb"))
                self.char2id = pickle.load(open(self.path + "/char2id", "rb"))
            return datas

        elif self.mode == "valid":
            datas = open(self.path + "/valid.txt", encoding="utf-8").read().strip("\n").splitlines()
            datas = [s.split() for s in datas]
            self.word2id = pickle.load(open(self.path + "/word2id", "rb"))
            self.char2id = pickle.load(open(self.path + "/char2id", "rb"))
            return datas

        elif self.mode == "test":
            datas = open(self.path + "/test.txt", encoding="utf-8").read().strip("\n").splitlines()
            datas = [s.split() for s in datas]
            self.word2id = pickle.load(open(self.path + "/word2id", "rb"))
            self.char2id = pickle.load(open(self.path + "/char2id", "rb"))
            return datas

    def generate_data_label(self, datas):
        char_datas = []
        weights = []
        for i, data in enumerate(datas):
            if i % 1000:
                print(i, len(datas))
            char_data = [[self.char2id["<start>"]] * self.max_word_length]
            for j, word in enumerate(data):
                char_list = []
                for char in word:
                    char_list.append(self.char2id.get(char, self.char2id["<unk>"]))
                char_list = char_list[0:self.max_word_length] + [self.char2id["<pad>"]] * (self.max_word_length
                                                                                           - len(char_list))

                # change the datas that contains words into word_id_list
                datas[i][j] = self.word2id.get(datas[i][j].lower(), self.word2id["<unk>"])
                char_data.append(char_list)
            weights.extend(len(datas[i]) * [1] + (self.max_sentence_length - len(datas[i])) * [0])
            datas[i] = datas[i][0:self.max_sentence_length] + [self.word2id["<pad>"]] * \
                       (self.max_sentence_length - len(datas[i]))

            char_datas.append(char_data)
            char_datas[i] = char_datas[i][0:self.max_sentence_length] + \
                            (self.max_sentence_length - len(char_datas[i])) * \
                            [[self.char2id["<pad>"]] * self.max_word_length]

        datas = np.array(datas)
        char_datas = np.array(char_datas)
        weights = np.array(weights)
        return datas, char_datas, weights


if __name__ == "__main__":
    CharLMDataSet()
