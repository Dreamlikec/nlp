import numpy as np
from collections import deque


class DataProcess(object):
    def __init__(self, input_file_name, min_count):
        self.input_file_name = input_file_name
        self.input_file = open(input_file_name, "r")
        self.min_count = min_count
        self.index = 0
        self.word2id_dict = dict()
        self.id2word_dict = dict()
        self.word_frequency_dict = dict()
        self.word_count_sum = 0  # total words number
        self.word_count = 0  # unique word number
        self.sample_table = []
        self.wordId_list = []
        self.word_pairs_queue = deque()
        self.sentence_count = 0  # total number of sentence
        self._init_dict()
        self._init_sample_table()
        self.get_wordId_list()

        print("Word Count is:", self.word_count)
        print("word Count Sum is:", self.word_count_sum)
        print("Sentence Count is:", self.sentence_count)

    def _init_dict(self):
        word_freq = dict()
        for line in self.input_file:
            line = line.strip().split(" ")
            self.word_count_sum += len(line)
            self.sentence_count += 1
            for word in line:
                if word_freq.get(word) is None:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        for word, freq in word_freq.items():
            if freq < self.min_count:
                self.word_count_sum -= freq
                continue
            self.word_frequency_dict[word] = word_freq[word]
            self.word2id_dict[word] = len(self.word2id_dict)
            self.id2word_dict[len(self.id2word_dict)] = word
        self.word_count = len(self.word2id_dict)

    def _init_sample_table(self):
        sample_table_size = 1e8
        pow_frequency = np.array(list(self.word_frequency_dict.values())) ** 0.75
        ratio_array = pow_frequency / pow_frequency.sum()
        word_count_list = np.round(ratio_array * sample_table_size).astype('int')
        for word_index, word_count in enumerate(word_count_list):
            self.sample_table += [word_index] * word_count
        self.sample_table = np.array(self.sample_table)

    def get_wordId_list(self):
        self.input_file = open(self.input_file_name)
        sentence = self.input_file.readline()
        wordId_list = []
        sentence = sentence.strip().split(" ")
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
                context_ids = []
                for i in range(max(self.index - window_size, 0),
                               min(self.index + window_size + 1, len(self.wordId_list))):
                    if self.index == i:
                        continue
                    context_ids.append(self.wordId_list[i])
                self.word_pairs_queue.append((context_ids, wordId_w))
                self.index += 1
        result_pairs = []
        for _ in range(batch_size):
            result_pairs.append(self.word_pairs_queue.popleft())
        return result_pairs

    def get_negative_sampling(self, pos_pairs, neg_count):
        neg_u = np.random.choice(self.sample_table, size=(len(pos_pairs), neg_count))
        return neg_u

    def evaluate_pairs_count(self, window_size):
        return self.word_count_sum * (2 * window_size - 1) - (self.sentence_count - 1) * (1 + window_size) * window_size


# 测试所有方法
def test():
    test_data = DataProcess('../data/text8.txt', 3)
    test_data.evaluate_pairs_count(2)
    pos_pairs = test_data.get_batch_pairs(10, 2)
    print('正采样:')
    print(pos_pairs)
    pos_word_pairs = []
    for pair in pos_pairs:
        pos_word_pairs.append(([test_data.id2word_dict[i] for i in pair[0]], test_data.id2word_dict[pair[1]]))
    print(pos_word_pairs)
    neg_pair = test_data.get_negative_sampling(pos_pairs, 3)
    print('负采样:')
    print(neg_pair)
    neg_word_pair = []
    for pair in neg_pair:
        neg_word_pair.append(
            (test_data.id2word_dict[pair[0]], test_data.id2word_dict[pair[1]], test_data.id2word_dict[pair[2]]))
    print(neg_word_pair)

if __name__ == '__main__':
    test()
