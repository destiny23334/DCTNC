import torch.nn as nn


class DNN512(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DNN512, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
