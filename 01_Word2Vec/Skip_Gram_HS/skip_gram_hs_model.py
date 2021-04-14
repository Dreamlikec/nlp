import torch
import torch.nn as nn
import torch.nn.functional as f


class SkipGramHSModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGramHSModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.w_embeddings = nn.Embedding(2 * vocab_size - 1, embed_size).cuda()
        self.v_embeddings = nn.Embedding(2 * vocab_size - 1, embed_size).cuda()

    def _init_embed(self):
        int_range = 0.5 / self.embed_size
        self.w_embeddings.weight.data.uniform_(-int_range, int_range)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_w, pos_v, neg_w, neg_v):
        pos_w_emb = self.w_embeddings(torch.LongTensor(pos_w).cuda())  # batch_size * embedding_size
        pos_v_emb = self.v_embeddings(torch.LongTensor(pos_v).cuda())  # batch_size * embedding_size
        neg_w_emb = self.w_embeddings(torch.LongTensor(neg_w).cuda())  # batch_size * embedding_size
        neg_v_emb = self.v_embeddings(torch.LongTensor(neg_v).cuda())  # batch_size * embedding_size

        score1 = torch.mul(pos_w_emb, pos_v_emb)
        score1 = torch.sum(score1, dim=1)
        score1 = torch.clamp(score1, min=-10, max=10)
        score1 = f.logsigmoid(score1)

        score2 = torch.mul(neg_w_emb, neg_v_emb)
        score2 = torch.sum(score2, dim=1)
        score2 = torch.clamp(score2, min=-10, max=10)
        score2 = f.logsigmoid(-score2)

        loss = -(score1.sum() + score2.sum()).mean()
        return loss

    def save_embeddings(self, id2word_dict, output_file_name):
        embedding = self.w_embeddings.weight.data.cpu().numpy()
        f_out = open(output_file_name, "w")
        f_out.write("%d %d\n" % (self.vocab_size, self.embed_size))
        for word_id, word in id2word_dict.items():
            e = embedding[word_id]
            e = " ".join(map(lambda x: str(x), e))
            f_out.write("%s %s\n" % (word, e))


def test():
    model = SkipGramHSModel(100, 10)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    pos_w = [0, 0, 1, 1, 1]
    pos_v = [1, 2, 0, 2, 3]
    neg_v = [23, 42, 32, 35, 24]
    model.forward(pos_w, pos_v, neg_v)
    model.save_embeddings(id2word, 'test.txt')


if __name__ == '__main__':
    test()
