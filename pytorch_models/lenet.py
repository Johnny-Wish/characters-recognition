import torch.nn as nn
from pytorch_models.torch_utils import EmbedModule, EmbedModuleBuilder
from torchvision.transforms import Compose
from preprocess import Reshape


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


class LeNetBuilder(EmbedModuleBuilder):
    def _instantiate_model(self):
        self._model = LeNet(self.num_channels, self.num_classes)

    def _process_state_dict(self, d):
        return d

    def _set_trainable(self):
        super(LeNetBuilder, self)._set_trainable()

        if not self.train_features:
            self._model.features.requires_grad = False
            for param in self._model.parameters():
                param.requires_grad = False


# alias for model getter along with default args and kwargs
builder_class = LeNetBuilder
model_args = ()
model_kwargs = dict(
    num_channels=1,
)

# dataset transformer corresponding to the model
transformer = Compose([
    Reshape(1, 28, 28),
])
