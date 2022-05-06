import torch
import torch.nn as nn


class serverModel(nn.Module):

    def __init__(self, num_classes=10, num_clients=2, dim=128):
        super(serverModel, self).__init__()
        self.fc = nn.Linear(dim * num_clients, num_classes)
        #self.classifier = nn.Sequential(
        #    nn.Dropout(),
        #    nn.Linear(256 * 6 * 6 * 4, 4096),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(4096, 4096),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(4096, num_classes),
        #)

    def forward(self, x):
        pooled_view = self.fc(x)
        return pooled_view
