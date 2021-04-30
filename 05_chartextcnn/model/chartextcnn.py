import torch
import torch.nn as nn
import torch.nn.functional as f
from torchsummary import summary


class charTextCNN(nn.Module):
    def __init__(self, config):
        super(charTextCNN, self).__init__()
        int_features = [config.char_num] + config.features[:-1]
        out_features = config.features
        kernel_sizes = config.kernel_sizes

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=int_features[0], out_channels=out_features[0], kernel_size=kernel_sizes[0], stride=1),
            nn.BatchNorm1d(out_features[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=int_features[1], out_channels=out_features[1], kernel_size=kernel_sizes[1], stride=1),
            nn.BatchNorm1d(out_features[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=int_features[2], out_channels=out_features[2], kernel_size=kernel_sizes[2], stride=1),
            nn.BatchNorm1d(out_features[2]),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=int_features[3], out_channels=out_features[3], kernel_size=kernel_sizes[3], stride=1),
            nn.BatchNorm1d(out_features[3]),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=int_features[4], out_channels=out_features[4], kernel_size=kernel_sizes[4], stride=1),
            nn.BatchNorm1d(out_features[4]),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=int_features[5], out_channels=out_features[5], kernel_size=kernel_sizes[5], stride=1),
            nn.BatchNorm1d(out_features[5]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 34, 1024),
            nn.ReLU(),
            nn.Dropout(p=config.dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=config.dropout)
        )
        self.fc3 = nn.Linear(1024, config.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class config:
    def __init__(self):
        self.char_num = 70  # 字符的个数
        self.features = [256, 256, 256, 256, 256, 256]  # 每一层特征个数
        self.kernel_sizes = [7, 7, 3, 3, 3, 3]  # 每一层的卷积核尺寸
        self.dropout = 0.5  # dropout大小
        self.num_classes = 4  # 数据的类别个数


if __name__ == "__main__":
    config = config()
    ctc = charTextCNN(config).cuda()
    print(summary(ctc, input_size=(70, 1014)))
