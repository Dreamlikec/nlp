import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
from data import AG_Data
from model import FastText
from config import ArgumentParser


def get_test_result(data_loader, device, length):
    model.eval()
    model.to(device)
    true_sample_num = 0
    for data, label in data_loader:
        data.to(device)
        label.to(device)
        test_out = model(data)
        true_sample_num += np.sum((torch.argmax(test_out, 1) == label).cpu().numpy())
    acc = true_sample_num / length
    return acc


config = ArgumentParser()
if __name__ == "__main__":
    device = torch.device("cpu")
    if config.cuda and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
        device = torch.device(config.cuda + config.gpu)

    train_set = AG_Data("/AG/train.csv", min_count=config.min_count, max_length=config.max_length,
                        n_gram=config.n_gram)
    train_loader = DataLoader(dataset=train_set, shuffle=True, num_workers=2, batch_size=config.batch_size)

    uniwords_num = train_set.uniwords_num
    word2id = train_set.word2id

    test_set = AG_Data("/AG/test.csv", min_count=config.min_count, max_length=config.max_length,
                       n_gram=config.n_gram, uniwords_num=uniwords_num, word2id=word2id)
    test_loader = DataLoader(dataset=test_set, shuffle=False, num_workers=2, batch_size=config.batch_size)

    criterion = nn.CrossEntropyLoss()

    model = FastText(vocab_size=uniwords_num + 100000, embedding_size=config.embed_size, max_length=config.max_length,
                     label_num=config.label_num)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss = -1

    model.train()
    model = model.to(device)
    for epoch in range(config.epoch):
        process_bar = tqdm(train_loader)
        for data, label in process_bar:
            data = data.to(device)
            label = label.to(device)

            out = model(data)
            loss_now = criterion(out, Variable(label.long()))
            l2_loss = torch.sum(torch.pow(list(model.parameters())[1], 2))
            loss_now = loss_now + 0.005 * l2_loss  # L2_norm

            if loss == -1:
                loss = loss_now.data.item()
            else:
                loss = 0.95 * loss + 0.05 * loss_now.data.item()
            process_bar.set_postfix(loss=loss)
            process_bar.update()
            optimizer.zero_grad()
            loss_now.backward()
            optimizer.step()
        test_acc = get_test_result(test_loader, device, length=len(test_set))
        print("The test acc is: %.5f" % test_acc)
