import torch.nn as nn

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='DRSN_CW', num_classes=None):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, features):
        return self.fc(features)
