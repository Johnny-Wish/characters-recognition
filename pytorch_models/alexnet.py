import torch
import torch.nn as nn
from torchvision.models.alexnet import model_urls
from torch.utils.model_zoo import load_url
from pytorch_models.torch_utils import EmbedModule


class AlexNet(EmbedModule):
    """an alexnet model with 1 or 3 input channels, which returns prediction logits (instead of probs)"""

    def __init__(self, num_channels=3, num_classes=1000):
        super(AlexNet, self).__init__()
        assert num_channels in [1, 3], "illegal input channels={}, must be either 1 or 3".format(num_channels)

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def embed(self, x):
        embeddings = self.features(x)
        flattened_embeddings = embeddings.view(x.size(0), 256 * 6 * 6)
        return flattened_embeddings


def get_alexnet(num_channels=3, num_classes=1000, pretrained=True, pretrained_path=None, train_features=True):
    model = AlexNet(num_channels=num_channels, num_classes=num_classes)

    if pretrained:
        if pretrained_path is None:
            pretrained_dict = load_url(model_urls['alexnet'])
        else:
            pretrained_dict = torch.load(pretrained_path)
        if num_channels == 1:
            pretrained_dict["features.0.weight"] = pretrained_dict["features.0.weight"].sum(dim=1, keepdim=True)
        pretrained_num_classes = len(pretrained_dict["classifier.6.weight"])
        if num_classes < pretrained_num_classes:  # The output layer is subject to changes for different tasks
            pretrained_dict["classifier.6.weight"] = pretrained_dict["classifier.6.weight"][:num_classes]
            pretrained_dict["classifier.6.bias"] = pretrained_dict["classifier.6.bias"][:num_classes]
        model.load_state_dict(pretrained_dict)

        if not train_features:
            model.features.requires_grad = False
            for param in model.features.parameters():
                param.requires_grad = False
    elif not train_features:
        print("The model is assigned random init weights. All layers must be trained")

    return model
