import torch
import torch.nn as nn
# from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet
from pytorch_models.torch_utils import EmbedModule


class ResNet3Channel(ResNet, EmbedModule):
    def __init__(self, block, layers, num_classes=1000):
        ResNet.__init__(self, block, layers, num_classes=num_classes)

    @property
    def features(self):
        return [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4,
                self.avgpool]

    def embed(self, input):
        for mapping in self.features:
            input = mapping(input)

        return input


class ResNet1Channel(ResNet3Channel):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet1Channel, self).__init__(block, layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)


architecture = {
    18: ([2, 2, 2, 2], BasicBlock),
    34: ([3, 4, 6, 3], BasicBlock),
    50: ([3, 4, 6, 3], Bottleneck),
    101: ([3, 4, 23, 3], Bottleneck),
    152: ([3, 8, 36, 3], Bottleneck),
}


def get_resnet(depth, num_channels=1, num_classes=62, pretrained_path=None, train_features=True):
    if num_channels == 1:
        Model = ResNet1Channel
    elif num_channels == 3:
        Model = ResNet3Channel
    else:
        raise ValueError("num_channels must be either 1 or 3")

    if depth not in architecture:
        raise ValueError("depth must be one of {}".format(architecture.keys()))

    layers, block = architecture[depth]
    model = Model(block, layers, num_classes=num_classes)

    if pretrained_path is not None:
        if pretrained_path == '':
            raise NotImplementedError("loading weights pretrained on ImageNet not implemented yet")
        else:
            state_dict = torch.load(pretrained_path)
            model.load_state_dict(state_dict)

        if not train_features:
            for feature in model.features:
                feature.requires_grad = False
                for param in feature.parameters():
                    param.requires_grad = False

    elif not train_features:
        print("The model is assigned random init weights. All layers must be trained")

    return model


get_model = get_resnet

model_args = ()
model_kwargs = {dict(
    depth=18,
    num_channels=1,
)}
