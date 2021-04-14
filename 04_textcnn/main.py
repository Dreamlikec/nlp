from pytorchtools import EarlyStopping
import torch
import torch.autograd as autograd
import warnings
import torch.nn as nn
import torch.optim as optim
from model import TextCNN
import numpy as np
import config
from data.MR_Dataset import MR_Dataset
from torch.utils.data import DataLoader

config = config.ArgumentParser()


def get_test_result(model, valid_data_iter, length):
    model.eval()
    model.to(config.device)
    valid_loss = 0
    true_sample_num = 0
    for data, label in valid_data_iter:
        data = data.to(config.device)
        label = label.to(config.device)
        out = model(data)
        loss = criterion(out, label.long())
        valid_loss += loss.data.item()
        true_sample_num += np.sum((torch.argmax(out, dim=1) == label).cpu().numpy())
    acc = true_sample_num / length
    return valid_loss, acc


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    acc = 0
    for i in range(0, 10):
        early_stopping = EarlyStopping(patience=10, verbose=True, cv_index=i)
        training_set = MR_Dataset(state="train", k=i, embedding_type="word2vec")
        if config.use_pretrained_embed:
            config.embeddings_pretrained = torch.from_numpy(training_set.embeddings_weights).float()
        else:
            config.embeddings_pretrained = False
        config.vocab_size = training_set.vocab_size
        print(config.vocab_size)
        training_iter = DataLoader(dataset=training_set,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   num_workers=2)

        valid_set = MR_Dataset(state="valid", k=i, embedding_type="no")
        valid_iter = DataLoader(dataset=valid_set,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=2)

        test_set = MR_Dataset(state="test", k=i, embedding_type="no")
        test_iter = DataLoader(dataset=test_set,
                               batch_size=config.batch_size,
                               shuffle=True,
                               num_workers=2)

        print(config.device)
        model = TextCNN(config).to(config.device)
        config.embeddings_pretrained.to(config.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        count = 0
        loss_sum = 0

        for epoch in range(config.epoch):
            model.train()
            for data, label in training_iter:
                data = data.to(config.device)
                label = label.to(config.device).squeeze()
                out = model(data)
                l2_loss = config.l2 * torch.pow(list(model.parameters())[1], 2).sum()
                loss = criterion(out, label.long()) + l2_loss

                loss_sum += loss.data.item()
                count += 1
                if count % 100 == 0:
                    print("Epoch", epoch, end=" ")
                    print("The loss is %.5f" % (loss_sum / 100))
                    loss_sum = 0
                    count = 0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            valid_loss, valid_acc = get_test_result(model, valid_iter, len(valid_set))
            early_stopping(valid_loss, model)
            print("The valid acc is %.5f" % valid_acc)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # the result of 1 epoch
        model.load_state_dict(torch.load('./checkpoints/checkpoint%d.pt' % i))
        test_loss, test_acc = get_test_result(model, test_iter, len(test_set))
        print("The test acc is: %.5f" % test_acc)
        acc += test_acc / 10
    print("The test acc is: %.5f" % acc)
