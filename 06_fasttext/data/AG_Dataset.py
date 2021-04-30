import numpy as np
import nltk
import os
import csv
from torch.utils.data import Dataset


class AG_Data(Dataset):
    def __init__(self, file_path, min_count, max_length, n_gram=2, word2id=None, uniwords_num=0):
        self.label = []
        self.data = []
        self.path = os.path.abspath(".")
        if "data" not in self.path:
            self.path += "/data"
        self.n_gram = n_gram
        self.load(file_path)
        if word2id is None:
            self.create_word2id(self.data, min_count)
        else:
            self.word2id = word2id
            self.uniwords_num = uniwords_num
        self.data = self.convert_data2id(self.data, max_length)
        self.data = np.array(self.data)
        self.label = np.array(self.label)

    def __getitem__(self, item):
        x = self.data[item]
        y = self.label[item]
        return x, y

    def __len__(self):
        return len(self.data)

    def load(self, data_path, lowercase=True):
        with open(self.path + data_path, "r") as f:
            datas = list(csv.reader(f, delimiter=",", quotechar='"'))
            for row in datas:
                self.label.append(int(row[0]) - 1)
                txt = " ".join(row[1:])
                if lowercase:
                    txt = txt.lower()
                txt = nltk.word_tokenize(txt)

                new_txt = []
                for i in range(len(txt)):
                    for j in range(self.n_gram):
                        if j <= i:
                            new_txt.append(" ".join(txt[i - j:i + 1]))
                self.data.append(new_txt)

    def create_word2id(self, datas, min_count=3):
        word_freq = {}
        for data in datas:
            for word in data:
                if word_freq.get(word) is not None:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

        word2id = {"<pad>": 0, "<unk>": 1}
        for word in word_freq:
            if word_freq[word] < min_count or " " in word:
                continue
            word2id[word] = len(word2id)  # uni_gram word
        self.uniwords_num = len(word2id)

        for word in word_freq:
            if word_freq[word] < min_count or " " not in word:
                continue
            word2id[word] = len(word2id)  # n_gram word
        self.word2id = word2id

    def convert_data2id(self, datas, max_length):
        for i, data in enumerate(datas):
            for j, word in enumerate(data):
                if " " not in word:
                    datas[i][j] = self.word2id.get(word, 1)
                else:
                    datas[i][j] = self.word2id.get(word, 1) % 100000 + self.uniwords_num
            datas[i] = datas[i][0:max_length] + [0] * (max_length - len(datas[i]))
        return datas


if __name__ == "__main__":
    ag_data = AG_Data("/AG/train.csv", 3, 100)
    print(ag_data.data.shape)
    print(ag_data.label.shape)
    print(ag_data.data[-20:])
    print(len(ag_data.data))
