import numpy as np
import sys

sys.path.append("../Skip_Gram_HS")
from collections import deque
from huffman_tree import HuffmanTree


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
        self.huffman_tree = HuffmanTree(self.wordId_frequency_dict)
        self.huffman_pos_path, self.huffman_neg_path = self.huffman_tree.get_all_pos_and_neg_path()
        self.word_pairs_queue = deque()

        self.get_wordId_list()
        print('Word Count is:', self.word_count)
        print('Word Count Sum is', self.word_count_sum)
        print('Sentence Count is:', self.sentence_count)
        print('Tree Node is:', len(self.huffman_tree.huffman))

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
                for i in range(max(self.index - window_size, 0),
                               min(self.index + window_size + 1, len(self.wordId_list))):
                    wordId_w = self.wordId_list[self.index]
                    wordId_v = self.wordId_list[i]
                    if self.index == i:
                        continue
                    self.word_pairs_queue.append((wordId_w, wordId_v))
                self.index += 1
        result_pairs = []
        for _ in range(batch_size):
            result_pairs.append(self.word_pairs_queue.popleft())
        return result_pairs

    def get_pairs(self, pos_pairs):
        neg_word_pair = []
        pos_word_pair = []
        for pair in pos_pairs:
            pos_word_pair += zip([pair[0]] * len(self.huffman_pos_path[pair[1]]), self.huffman_pos_path[pair[1]])
            neg_word_pair += zip([pair[0]] * len(self.huffman_neg_path[pair[1]]), self.huffman_neg_path[pair[1]])
        return pos_word_pair, neg_word_pair

    def evaluate_pairs_count(self, window_size):
        length = self.word_count_sum * (2 * window_size - 1) - (self.sentence_count - 1) * (1 + window_size) * window_size
        return length


def test():
    test_data = DataProcess('../data/text8.txt', 3)

    test_data.evaluate_pairs_count(2)
    pos_pairs = test_data.get_batch_pairs(10, 2)
    print(pos_pairs)
    pos_word_pairs = []
    for pair in pos_pairs:
        pos_word_pairs.append((test_data.id2word_dict[pair[0]], test_data.id2word_dict[pair[1]]))
    print(pos_word_pairs)
    pos_word_pair, neg_word_pair = test_data.get_pairs(pos_pairs)
    print(pos_word_pair, neg_word_pair)


if __name__ == '__main__':
    test()
