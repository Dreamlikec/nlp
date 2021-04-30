from torch.utils.data import Dataset
import os
import csv
import json
import numpy as np


class AG_Data(Dataset):
    def __init__(self, data_path, l0=1014):
        self.path = os.path.abspath(".")
        if "data" not in self.path:
            self.path += "/data"
        self.alphabet = ""
        self.data_path = data_path
        self.y = []
        self.data = []
        self.label = []
        self.load_Alphabet()
        self.load(self.data_path)
        self.l0 = l0

    def __getitem__(self, item):
        X = self.onHotEncode(item)
        y = self.y[item]
        return X, y

    def __len__(self):
        return len(self.label)

    def load_Alphabet(self):
        with open(self.path + "/alphabet.json", 'r') as f:
            self.alphabet = "".join(json.load(f))

    def load(self, data_path, lowercase=True):
        with open(self.path + data_path, "r") as f:
            datas = list(csv.reader(f, delimiter=",", quotechar='"'))
            for row in datas:
                self.label.append(int(row[0]) - 1)
                txt = " ".join(row[1:])
                if lowercase:
                    txt.lower()
                self.data.append(txt)
        self.y = self.label

    def onHotEncode(self, idx):
        X = np.zeros((self.l0, len(self.alphabet)))
        for char_idx, char in enumerate(self.data[idx]):
            if self.char2Index(char) != -1:
                X[char_idx][self.char2Index(char)] = 1.0
        return X.T

    def char2Index(self, char):
        return self.alphabet.find(char)


if __name__ == "__main__":
    ag = AG_Data(data_path="/AG/train.csv")
