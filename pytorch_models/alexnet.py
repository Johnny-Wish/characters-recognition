import torch
import torch.nn as nn
from torchvision.models.alexnet import model_urls
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor
from preprocess import Reshape
from torch.utils.model_zoo import load_url
from pytorch_models.torch_utils import EmbedModule, EmbedModuleBuilder


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


# TODO convert the following getter class to a subclass of EmbedModuleBuilder
class AlexNetBuilder(EmbedModuleBuilder):
    def _set_trainable(self):
        super(AlexNetBuilder, self)._set_trainable()

        if not self.train_features:
            self._model.features.requires_grad = False
            for param in self._model.features.parameters():
                param.requires_grad = False

    def _process_state_dict(self, d):
        if self.num_channels == 1:
            d["features.0.weight"] = d["features.0.weight"].sum(dim=1, keepdim=True)
        pretrained_num_classes = len(d["classifier.6.weight"])
        if self.num_classes < pretrained_num_classes:
            d["classifier.6.weight"] = d["classifier.6.weight"][:self.num_classes]
            d["classifier.6.bias"] = d["classifier.6.bias"][:self.num_classes]

        return d

    def _get_state_dict(self):
        if self.pretrained_path == "":
            return load_url(model_urls['alexnet'], map_location=self.device)
        else:
            return torch.load(self.pretrained_path, map_location=self.device)

    def _instantiate_model(self):
        return AlexNet(self.num_channels, self.num_classes)


# alias for model getter along with default args and kwargs
builder_class = AlexNetBuilder
model_args = ()
model_kwargs = dict(
    num_channels=1,
)

# dataset transformer corresponding to the model
transformer = Compose([
    Reshape(28, 28, 1),
    ToPILImage(),
    Resize((227, 227)),
    ToTensor(),
])
