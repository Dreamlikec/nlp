import numpy as np
import sys

sys.path.append("../Skip_Gram_HS")
from collections import deque


class DataProcess:
    def __init__(self, input_file_name, min_count):
        self.input_file_name = input_file_name
        self.index = 0
        self.input_file = open(self.input_file_name)
        self.min_count = min_count
        self.wordId_frequency_dict = dict()
        self.word_count = 0
        self.word_count_sum = 0
        self.sentence_count = 0
        self.id2word_dict = dict()
        self.word2id_dict = dict()
        self._init_dict()
        self.sample_table = []
        self._init_sample_table()  # negative sampling table initialize
        self.get_wordId_list()
        self.word_pairs_queue = deque()
        print('Word Count is:', self.word_count)
        print('Word Count Sum is', self.word_count_sum)
        print('Sentence Count is:', self.sentence_count)

    def _init_dict(self):
        word_freq = dict()
        for line in self.input_file:
            line = line.strip().split(' ')
            self.word_count_sum += len(line)
            self.sentence_count += 1
            for i, word in enumerate(line):
                if i % 1000000 == 0:
                    print(i, len(line))
                try:
                    word_freq[word] += 1
                except:
                    word_freq[word] = 1
        word_id = 0

        for per_word, per_count in word_freq.items():
            if per_count < self.min_count:  # 去除低频
                self.word_count_sum -= per_count
                continue
            self.id2word_dict[word_id] = per_word
            self.word2id_dict[per_word] = word_id
            self.wordId_frequency_dict[word_id] = per_count
            word_id += 1
        self.word_count = len(self.word2id_dict)

    def _init_sample_table(self):
        sample_table_size = 1e8
        pow_frequecy = np.array(list(self.wordId_frequency_dict.values())) ** 0.75
        word_pow_sum = np.sum(pow_frequecy)
        ratio_array = pow_frequecy / word_pow_sum
        word_count_list = np.round(ratio_array * sample_table_size).astype("int")
        for word_index, word_frequency in enumerate(word_count_list):
            self.sample_table += [word_index] * word_frequency
        self.sample_table = np.array(self.sample_table)
        np.random.shuffle(self.sample_table)

    def get_wordId_list(self):
        self.input_file = open(self.input_file_name, encoding="utf-8")
        sentence = self.input_file.readline()
        wordId_list = []  # 一句中的所有word 对应的 id
        sentence = sentence.strip().split(' ')
        for i, word in enumerate(sentence):
            if i % 1000000 == 0:
                print(i, len(sentence))
            try:
                word_id = self.word2id_dict[word]
                wordId_list.append(word_id)
            except:
                continue
        self.wordId_list = wordId_list

    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pairs_queue) < batch_size:
            for _ in range(1000):
                if self.index == len(self.wordId_list):
                    self.index = 0
                wordId_w = self.wordId_list[self.index]
                for i in range(max(self.index - window_size, 0),
                               min(self.index + window_size + 1, len(self.wordId_list))):

                    wordId_v = self.wordId_list[i]
                    if self.index == i:
                        continue
                    self.word_pairs_queue.append((wordId_w, wordId_v))
                self.index += 1
        result_pairs = []  # 返回mini-batch大小的正采样对
        for _ in range(batch_size):
            result_pairs.append(self.word_pairs_queue.popleft())
        return result_pairs

    def get_negative_sampling(self, positive_pairs, neg_count):
        neg_v = np.random.choice(self.sample_table, size=(len(positive_pairs), neg_count))
        return neg_v

    def evaluate_pairs_count(self, window_size):
        length = self.word_count_sum * (2 * window_size - 1) - (self.sentence_count - 1) * (
                1 + window_size) * window_size
        return length


if __name__ == "__main__":
    test_data = DataProcess('../data/text8.txt', 1)
    test_data.evaluate_pairs_count(2)
    pos_pairs = test_data.get_batch_pairs(10, 2)
    print('正采样:')
    print(pos_pairs)
    pos_word_pairs = []
    for pair in pos_pairs:
        pos_word_pairs.append((test_data.id2word_dict[pair[0]], test_data.id2word_dict[pair[1]]))
    print(pos_word_pairs)
    neg_pair = test_data.get_negative_sampling(pos_pairs, 3)
    print('负采样:')
    print(neg_pair)
    neg_word_pair = []
    for pair in neg_pair:
        neg_word_pair.append(
            (test_data.id2word_dict[pair[0]], test_data.id2word_dict[pair[1]], test_data.id2word_dict[pair[2]]))
    print(neg_word_pair)
