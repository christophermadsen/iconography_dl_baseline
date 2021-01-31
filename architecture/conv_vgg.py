import torch.nn as nn
import torch.nn.functional as F
from architecture import vgg
import torch

# import itertools
# state_dict = dict(itertools.islice(state_dict.items(), 10))

class ConvVGG(nn.Module):
    def __init__(self, num_classes):
        super(ConvVGG, self).__init__()

        vgg16_bn = vgg.vgg16_bn(num_classes=num_classes)
        self.num_classes = num_classes

        # New model is vgg16_bn until final conv2 and batchnorm
        self.new_features = vgg16_bn.features[:43]
        self.new_classifier = nn.Conv2d(2048, num_classes, 1, bias=False)


    def forward(self, x):
        x = self.new_features(x)
        x = self.gap2d(x, keepdims=True)
        x = self.new_classifier(x)
        x = x.view(-1, self.num_classes)
        return x

    def gap2d(self, x, keepdims=False):
        out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
        if keepdims:
            out = out.view(out.size(0), out.size(1), 1, 1)
        return out

class CAM(ConvVGG):
    def __init__(self, num_classes):
        super(CAM, self).__init__(num_classes)

    def forward(self, x):
        x = self.new_features(x)
        x = F.conv2d(x, self.new_classifier.weight)
        x = F.relu(x)
        return x
