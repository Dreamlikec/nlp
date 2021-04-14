import torch
import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        """
        load the model parameters
        """
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """
        save the model parameters
        """
        torch.save(self.state_dict(), path)

    def forward(self):
        pass


if __name__ == "__main__":
    model = BasicModule()
