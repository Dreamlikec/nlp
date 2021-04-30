import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config
from torch.utils.data import DataLoader
from model import charTextCNN
from data import AG_Data
from tqdm import tqdm

config = config.ArgumentParser()


def get_test_result(data_iter, length):
    model.eval()
    data_loss = 0
    true_sample_num = 0
    for data, label in data_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.float().cuda()
            label = label.long().cuda()
        else:
            data = torch.Tensor(data).float()
            label = torch.Tensor(label).long()
        out = model(data)
        loss = criterion(out, label)
        data_loss += loss.data.item()
        true_sample_num += np.sum((torch.argmax(out, 1) == label).cpu().numpy())
    acc = true_sample_num / length
    return data_loss, acc


if __name__ == "__main__":
    if config.gpu and torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)

    config.features = list(map(int, config.features.split(",")))
    config.kernel_sizes = list(map(int, config.kernel_sizes.split(",")))
    config.pooling = list(map(int, config.pooling.split(",")))

    train_set = AG_Data(data_path="/AG/train.csv", l0=config.l0)
    train_iter = DataLoader(dataset=train_set,
                            batch_size=config.batch_size,
                            shuffle=True,
                            num_workers=2)

    test_set = AG_Data(data_path="/AG/test.csv", l0=config.l0)
    test_iter = DataLoader(dataset=test_set,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=2)

    model = charTextCNN(config)
    if torch.cuda.is_available() and config.cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss = -1
    for epoch in range(config.num_epoch):
        model.train()
        process_bar = tqdm(train_iter)
        for data, label in process_bar:
            if config.cuda and torch.cuda.is_available():
                data = data.cuda().float()
                label = label.cuda().long()
            else:
                data = torch.Tensor(data).float()
                label = torch.Tensor(label).long().squeeze()
            out = model(data)
            loss_now = criterion(out, label)
            if loss == -1:
                loss = loss_now.data.item()
            else:
                loss = 0.95 * loss + 0.05 * loss_now.data.item()
            process_bar.set_postfix(loss=loss)
            process_bar.update()
            optimizer.zero_grad()
            loss_now.backward()
            optimizer.step()
        test_loss, test_acc = get_test_result(test_iter, len(test_set))
        print("The test acc is: %.5f" % test_acc)
