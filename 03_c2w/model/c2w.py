# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


class C2W(nn.Module):
    def __init__(self, c2w_config):
        super(C2W, self).__init__()
        self.char_hidden_size = c2w_config.char_hidden_size
        self.word_embed_size = c2w_config.word_embed_size
        self.char_embed_size = c2w_config.char_embed_size
        self.n_chars = c2w_config.n_chars
        self.char_embeddings = nn.Embedding(self.n_chars, self.char_embed_size)
        self.lm_hidden_size = c2w_config.lm_hidden_size
        self.vocab_size = c2w_config.vocab_size
        self.sentence_length = c2w_config.max_sentence_length
        self.char_lstm = nn.LSTM(input_size=self.char_embed_size, hidden_size=self.char_hidden_size, bidirectional=True,
                                 batch_first=True)
        self.lm_lstm = nn.LSTM(input_size=self.word_embed_size, hidden_size=self.lm_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(2 * self.char_hidden_size, self.word_embed_size)
        self.fc2 = nn.Linear(self.lm_hidden_size, self.vocab_size)

    def forward(self, x):
        x = self.char_embeddings(x)
        char_lstm_result = self.char_lstm(x)

        forward_last = char_lstm_result[0][:, -1, 0:self.char_hidden_size]
        reverse_first = char_lstm_result[0][:, 0, self.char_hidden_size:]
        word_input = torch.cat([forward_last, reverse_first], dim=1)

        word_input = self.fc1(word_input)
        word_input = word_input.view([-1, self.sentence_length, self.word_embed_size])
        lm_lstm_result = self.lm_lstm(word_input)[0].contiguous()
        lm_lstm_result = lm_lstm_result.view([-1, self.lm_hidden_size])
        out = self.fc2(lm_lstm_result)
        return out


class config:
    def __init__(self):
        self.n_chars = 64
        self.char_embed_size = 50
        self.max_sentence_length = 8
        self.char_hidden_size = 50
        self.lm_hidden_size = 150
        self.word_embed_size = 50
        self.vocab_size = 1000


if __name__ == "__main__":
    config = config()
    c2w = C2W(config)
    test = np.zeros([64, 16])
    test = torch.from_numpy(test).long()
    c2w(test)
