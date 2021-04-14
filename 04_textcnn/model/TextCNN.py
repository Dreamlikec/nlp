import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from model.BasicModule import BasicModule


class TextCNN(BasicModule):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        # embeddings layer
        self.device = config.device
        if config.embeddings_pretrained is not None:
            self.embeddings = nn.Embedding.from_pretrained(config.embeddings_pretrained, freeze=False).to(self.device)
        else:
            self.embeddings = nn.Embedding(config.vocab_size, config.embedding_size).to(self.device)

        # convolution layers

        self.convs = [
            nn.Conv1d(in_channels=config.embedding_size, out_channels=config.out_number, kernel_size=kernel_size).to(
                self.device) for kernel_size in config.kernel_sizes]

        # DropOut
        self.dropout = nn.Dropout(config.dropout).to(self.device)

        # classification layer
        self.fc = nn.Linear(config.out_number * len(config.kernel_sizes), config.label_num).to(self.device)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        # max_pooling
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = x.long()
        x = self.embeddings(x)  # batch_size * length * embedding_size
        x = x.transpose(1, 2).contiguous()  # batch_size * embedding_size * length
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class config:
    def __init__(self):
        self.embeddings_pretrained = None  # 是否使用预训练的词向量
        self.vocab_size = 100  # 词表中单词的个数
        self.embedding_size = 300  # 词向量的维度
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.out_number = 100  # 每种尺寸卷积核的个数
        self.kernel_sizes = [3, 4, 5]  # 卷积核的尺寸
        self.label_num = 2  # 标签个数
        self.dropout = 0.5  # dropout的概率
        self.sentence_max_size = 50  # 最大句子长度


if __name__ == "__main__":
    config = config()
    textcnn = TextCNN(config)
    print(summary(textcnn, input_size=(50,)))
