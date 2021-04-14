import torch
import torch.nn as nn


class Glove(nn.Module):
    def __init__(self, vocab_size, embed_size, x_max, alpha):
        super(Glove, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.x_max = x_max
        self.alpha = alpha
        self.w_embed = nn.Embedding(self.vocab_size, self.embed_size).type(torch.float64)
        self.w_bias = nn.Embedding(self.vocab_size, 1).type(torch.float64)
        self.v_embed = nn.Embedding(self.vocab_size, self.embed_size).type(torch.float64)
        self.v_bias = nn.Embedding(self.vocab_size, 1).type(torch.float64)

    def forward(self, w_data, v_data, labels):
        w_weights_embed = self.w_embed(w_data)
        w_bias_embed = self.w_bias(w_data)
        v_weights_embed = self.v_embed(v_data)
        v_bias_embed = self.v_bias(v_data)

        fx = torch.pow(labels / self.x_max, self.alpha)
        fx[fx > 1] = 1
        loss = (fx * ((w_weights_embed * v_weights_embed).sum(dim=1) + w_bias_embed + v_bias_embed - torch.log(labels))
                .pow(2)).mean()
        return loss

    def save_embeddings(self, word2id, file_name):
        embedding_1 = self.w_embed.cpu().numpy()
        embedding_2 = self.v_embed.cpu().numpy()
        embedding = (embedding_1 + embedding_2) / 2
        fout = open(file_name, 'w')
        fout.write("%d %d \n" % (len(word2id), self.embed_size))
        for word, wordid in word2id.items():
            e = embedding[wordid]
            e = " ".join(map(lambda x: str(x), e))
            fout.write("%s %s\n" % (word, e))
