import torch.nn as nn
from torchvision.models.alexnet import model_urls
from torch.utils.model_zoo import load_url


class AlexNet(nn.Module):
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


def get_alexnet(num_channels=3, num_classes=1000, pretrained=True):
    model = AlexNet(num_channels=num_channels, num_classes=num_classes)

    if pretrained:
        pretrained_dict = load_url(model_urls['alexnet'])
        if num_channels == 1:
            pretrained_dict["features.0.weight"] = pretrained_dict["features.0.weight"].sum(dim=1, keepdim=True)
        if num_classes < 1000:  # The output layer is subject to major changes for different tasks
            pretrained_dict["classifier.6.weight"] = pretrained_dict["classifier.6.weight"][:num_classes]
            pretrained_dict["classifier.6.bias"] = pretrained_dict["classifier.6.bias"][:num_classes]
        model.load_state_dict(pretrained_dict)

    return model
