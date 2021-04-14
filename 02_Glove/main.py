from data import Wiki_Dataset
from model import Glove
from tqdm import tqdm
from torch.utils import data
import torch
import torch.optim as optim
import numpy as np
import config as argumentparser

if __name__ == "__main__":
    config = argumentparser.ArgumentParser()

    wiki_dataset = Wiki_Dataset(min_count=config.min_count, window_size=config.window_size)
    train_iter = data.DataLoader(dataset=wiki_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=2)
    model = Glove(vocab_size=len(wiki_dataset.word2id), embed_size=config.embed_size, x_max=config.x_max,
                  alpha=config.alpha)

    if config.cuda and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss = -1

    for epoch in range(config.epoch):
        process_bar = tqdm(train_iter)
        for data, label in process_bar:
            w = torch.tensor(np.array([sample[0] for sample in data]), dtype=torch.long)
            v = torch.tensor(np.array([sample[1] for sample in data]), dtype=torch.long)
            if config.cuda and torch.cuda.is_available():
                torch.cuda.set_device(config.gpu)
                w = w.cuda()
                v = v.cuda()
                label = label.cuda()
            loss_now = model(w, v, label)
            if loss == -1:
                loss = loss_now.data.item()
            else:
                loss = 0.95 * loss + 0.05 * loss_now.data.item()
            process_bar.set_postfix(loss=loss)
            process_bar.update()
            optimizer.zero_grad()
            loss_now.backward()
            optimizer.step()
    model.save_embeddings(wiki_dataset.word2id, "./embeddings/results.txt")
