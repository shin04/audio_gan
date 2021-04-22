import torch
import torch.nn as nn
from torchlibrosa.augmentation import SpecAugmentation

from .Conformer import ConformerBlock


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
                      stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(1, stride=1),
        ])

        self.fcl = nn.Linear(128*44, 1024)
        self.dp_layer = nn.Dropout(p=0.2)

        self.downsample_blocks = nn.ModuleList([
            ConformerBlock(dim=256),
            ConformerBlock(dim=256),
            ConformerBlock(dim=256),
        ])
        self.head = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # print("input shape: ", input.shape)
        augmented_data = self.specAug(input)
        # print("aug data shape: ", augmented_data.shape)

        for l in self.conv_layers:
            x = l(augmented_data)
            # print("x shape: ", x.shape)

        x = x.view(x.size()[0], -1)
        # print("x shape: ", x.shape)
        x = self.fcl(x)
        # print("x shape: ", x.shape)
        x = self.dp_layer(x)
        # print("x shape: ", x.shape)

        x = x.view(-1, 4, 256)
        for index, blk in enumerate(self.downsample_blocks):
            x = blk(x)
            # print("x shape: ", x.shape)

        x = x.view(x.size()[0], -1)
        x = self.head(x)
        output = self.sigmoid(x)

        return output

    def specAug(self, x):
        '''
        spwcAug augmentate data with SpecAugmentation.
        This returns torch tensor.

        args
        x: ndarray or torch tensor
        '''
        augmenter = SpecAugmentation(
            time_drop_width=32,
            time_stripes_num=2,
            freq_drop_width=32,
            freq_stripes_num=4
        )
        augmented_data = augmenter(x)

        return augmented_data
