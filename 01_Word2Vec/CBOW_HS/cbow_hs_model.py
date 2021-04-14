import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class CBowHSModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CBowHSModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.u_embeddings = nn.Embedding(2 * vocab_size - 1, embed_size, sparse=True).cuda()
        self.w_embeddings = nn.Embedding(2 * vocab_size - 1, embed_size, sparse=True).cuda()
        self._init_embed()

    def _init_embed(self):
        int_range = 0.5 / self.embed_size
        self.u_embeddings.weight.data.uniform_(-int_range, int_range)
        self.w_embeddings.weight.data.uniform_(-0, 0)

    def compute_context_matrix(self, u):
        pos_u_embed = []
        for per_Xw in u:
            per_u_emb = self.u_embeddings(torch.LongTensor(per_Xw).cuda())
            per_u_numpy = per_u_emb.data.cpu().numpy()
            per_u_numpy = np.sum(per_u_numpy, axis=0)
            per_u_list = per_u_numpy.tolist()
            pos_u_embed.append(per_u_list)
        pos_u_embed = torch.FloatTensor(pos_u_embed).cuda()
        return pos_u_embed

    def forward(self, pos_u, pos_w, neg_u, neg_w):
        pos_u_embed = self.compute_context_matrix(pos_u)
        pos_w_embed = self.w_embeddings(torch.LongTensor(pos_w).cuda())
        neg_u_embed = self.compute_context_matrix(neg_u)
        neg_w_embed = self.w_embeddings(torch.LongTensor(neg_w).cuda())

        score1 = torch.mul(pos_u_embed, pos_w_embed)
        score1 = torch.sum(score1, dim=1)
        score1 = f.logsigmoid(score1)
        score2 = torch.mul(neg_u_embed, neg_w_embed)
        score2 = torch.sum(score2, dim=1)
        score2 = f.logsigmoid(-score2)
        loss = (score1.sum() + score2.sum()).sum()
        return -loss

    def save_embeddings(self, id2word_dict, file_name):
        embedding = self.u_embeddings.weight.data.cpu().numpy()
        file_out = open(file_name, "w")
        file_out.write("%d %d\n" % (self.vocab_size, self.embed_size))
        for word_id, word in id2word_dict.items():
            e = embedding[word_id]
            e = " ".join(map(lambda x: str(x), e))
            file_out.write("%s %s\n" % (word, e))


def test():
    model = CBowHSModel(100, 10)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)

    pos_pairs = [([1, 2, 3], 50), ([0, 1, 2, 3], 70)]
    neg_pairs = [([1, 2, 3], 50), ([0, 1, 2, 3], 70)]
    pos_u = [pair[0] for pair in pos_pairs]
    pos_w = [int(pair[1]) for pair in pos_pairs]
    neg_u = [pair[0] for pair in neg_pairs]
    neg_w = [int(pair[1]) for pair in neg_pairs]
    model.forward(pos_u, pos_w, neg_u, neg_w)
    model.save_embeddings(id2word, 'test.txt')


if __name__ == '__main__':
    test()
