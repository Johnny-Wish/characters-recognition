import torch.nn as nn
from pytorch_models.torch_utils import EmbedModule, EmbedModuleBuilder


class MLP(EmbedModule):
    def __init__(self, num_channels, num_classes):
        if num_channels != 1:
            raise ValueError("num_channels must be 1 for {} to work".format(self.__class__.__name__))

        super(MLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 1500),
            nn.Sigmoid(),
            nn.Linear(1500, 600),
            nn.Sigmoid(),
            nn.Linear(600, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)

    def embed(self, x):
        embeddings = self.features(x)
        flattened_embeddings = embeddings.view(x.size(0), 50 * 7 * 7)
        return flattened_embeddings


class MLPBuilder(EmbedModuleBuilder):
    def _instantiate_model(self):
        self._model = MLP(num_channels=self.num_channels, num_classes=self.num_classes)

    def _process_state_dict(self, d):
        return d

    def _set_trainable(self):
        super(MLPBuilder, self)._set_trainable()


# alias for model getter along with default args and kwargs
builder_class = MLPBuilder
model_args = ()
model_kwargs = dict(
    num_channels=1,
)

# dataset transformer corresponding to the model
transformer = None
