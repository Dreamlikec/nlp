import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import CharLMDataSet
from model import C2W
from tqdm import tqdm
from config import ArgumentParser

config = ArgumentParser()


def get_test_result(data_iter, test_data):
    model.eval()
    all_ppl = 0
    for data, label, weights in data_iter:
        if config.cuda and torch.cuda.is_available():
            data = torch.tensor(data, dtype=torch.long).cuda()
            label = torch.tensor(label, dtype=torch.long).cuda()
            weights = torch.tensor(weights, dtype=torch.long).cuda()
        else:
            data = torch.tensor(data, dtype=torch.long)
            label = torch.tensor(label, dtype=torch.long)
            weights = torch.tensor(weights, dtype=torch.long)

        criterion = nn.CrossEntropyLoss(reduce=None)
        out = model(data)
        loss_now = criterion(out, label)
        ppl = (loss_now * weights.float()).view([-1, config.max_sentence_length])
        ppl = torch.sum(ppl, dim=1) / torch.sum(weights.view([-1, config.max_sentence_length]) != 0, dim=1).float()
        ppl = ppl.exp().sum()
        all_ppl += ppl
    return all_ppl * config.max_sentence_length / test_data.__len__()


if __name__ == "__main__":
    if config.cuda and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
    train_set = CharLMDataSet(mode="train")
    train_iter = DataLoader(dataset=train_set,
                            batch_size=config.batch_size * config.max_sentence_length,
                            shuffle=False,
                            num_workers=2)

    valid_set = CharLMDataSet(mode="valid")
    valid_iter = DataLoader(dataset=valid_set,
                            batch_size=config.batch_size * config.max_sentence_length,
                            shuffle=False,
                            num_workers=0)

    test_set = CharLMDataSet(mode="test")
    test_iter = DataLoader(dataset=valid_set,
                           batch_size=config.batch_size * config.max_sentence_length,
                           shuffle=False,
                           num_workers=0)

    model = C2W(config)
    if config.cuda and torch.cuda.is_available():
        model.cuda()
        print("training on gpu")
    criterion = nn.CrossEntropyLoss(reduce=None)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss = -1
    for epoch in range(config.epoch):
        model.train()
        process_bar = tqdm(train_iter)
        for data, label, weights in process_bar:
            if config.cuda and torch.cuda.is_available():
                data = torch.tensor(data, dtype=torch.long).cuda()
                label = torch.tensor(label, dtype=torch.long).cuda()
                weights = torch.tensor(weights, dtype=torch.long).cuda()
            else:
                data = torch.tensor(data, dtype=torch.long)
                label = torch.tensor(label, dtype=torch.long)
                weights = torch.tensor(weights, dtype=torch.long)

            out = model(data)
            loss_now = criterion(out, label)
            ppl = (loss_now * weights.float()).view(-1, config.max_sentence_length)
            a = ppl.cpu().data.numpy()
            ppl = torch.sum(ppl, dim=1) / torch.sum(weights.view([-1, config.max_sentence_length]) != 0, dim=1).float()
            ppl = ppl.exp().mean()

            loss_now = torch.sum(loss_now * weights.float()) / torch.sum(weights != 0)
            if loss == -1:
                loss = loss_now.data.item()
            else:
                loss = 0.9 * loss + 0.1 * loss_now
            process_bar.set_postfix(loss=ppl.data.item())
            process_bar.update()
            optimizer.zero_grad()
            loss_now.backward()
            optimizer.step()

    print("Valid ppl is:", get_test_result(valid_iter, valid_set))
    print("Test ppl is:", get_test_result(test_iter, test_set))
