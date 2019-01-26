import torch
import torch.nn as nn
from torch_utils import EmbedModule


class LeNet(EmbedModule):
    def __init__(self, num_channels, num_classes):
        super(LeNet, self).__init__()

        self.features = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.Conv2d(num_channels, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.ReplicationPad2d(2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(50 * 7 * 7, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 50 * 7 * 7)
        return self.classifier(x)

    def embed(self, x):
        embeddings = self.features(x)
        flattened_embeddings = embeddings.view(x.size(0), 50 * 7 * 7)
        return flattened_embeddings


def get_lenet(num_channels, num_classes, pretrained_path=None, train_features=True):
    model = LeNet(num_channels, num_classes)

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict)

        if not train_features:
            model.features.requires_grad = False
            for param in model.features.parameters():
                param.requires_grad = False

    elif not train_features:
        print("The model is assigned random init weights. All layers must be trained")

    return model
