import torch.nn as nn


cfgs = {
    'A':      [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'A-LRN':  [64, 'LRN', 'M', 128, 'LRN', 'M', 256, 'M', 512, 'M', 512, 'M'],
    'B':      [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C':      [64, 64, 'M', 128, 128, 'M', 256, 256, 1, 'M', 512, 512, 1, 'M', 512, 512, 1, 'M'],  # 1 means 1x1 conv
    'D':      [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E':      [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}



class VGG(nn.Module):
    def __init__(self, config_name='A', num_classes=35):
        super().__init__()
        self.features = self._make_layers(cfgs[config_name])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'LRN':
                layers += [nn.LocalResponseNorm(size=5)]
            elif v == 1:
                layers += [nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x


