import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_length, label_num):
        super(FastText, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.label_num = label_num

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.avg = nn.AvgPool1d(kernel_size=max_length, stride=1)
        self.fc = nn.Linear(embedding_size, label_num)

    def forward(self, x):
        x = self.embedding(x)  # batch_size * length * embedding_size
        x = x.permute(0, 2, 1).contiguous()
        x = self.avg(x).squeeze()
        x = self.fc(x)
        return x


if __name__ == "__main__":
    fasttext = FastText(vocab_size=1000, embedding_size=200, max_length=100, label_num=4)
    print(fasttext.parameters)
    state = torch.Tensor(torch.rand(64, 100)).long()
    out = fasttext(state)
    # print(summary(fasttext, input_size=(1, 100)))
    print(list(fasttext.parameters())[1].shape)
    print(out.size())
