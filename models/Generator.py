import torch
import torch.nn as nn

import math
import sys
import time

from conformer import ConformerBlock


class Generator(nn.Module):
    def __init__(self, latent_dim=100, embed_dim=16, height=int(128/4), width=int(44/4)):
        '''
        latent_dim: dimensionality of the latent space
        embed_dim: The base channel num of Generator
        width: melspec width/4
        height: melspec height/4
        '''
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.width = width
        self.height = height
        self.embed_dim = embed_dim

        self.l1 = nn.Linear(latent_dim, (width*height) * embed_dim)

        self.blocks = nn.ModuleList([
            ConformerBlock(dim=embed_dim),
            # ConformerBlock(dim=embed_dim),
            # ConformerBlock(dim=embed_dim),
            # ConformerBlock(dim=embed_dim),
            # ConformerBlock(dim=embed_dim),
        ])

        self.upsample_blocks = nn.ModuleList([
            nn.ModuleList([
                ConformerBlock(dim=embed_dim//4),
                # ConformerBlock(dim=embed_dim//4),
                # ConformerBlock(dim=embed_dim//4),
                # ConformerBlock(dim=embed_dim//4),
            ]),
            nn.ModuleList([
                ConformerBlock(dim=embed_dim//16),
                # ConformerBlock(dim=embed_dim//16),
            ])
        ])

        self.deconv = nn.Sequential(
            nn.Conv2d(embed_dim//16, 1, 1, 1, 0)
        )

    def forward(self, z):
        print("input: ", z.shape)
        s = time.time()
        x = self.l1(z)
        print(time.time() - s)
        print("l1:", x.shape)
        s = time.time()
        x = x.view(-1, self.height*self.width, self.embed_dim)
        print(time.time() - s)
        print("l1 -> reshape:", x.shape)
        B = x.size()
        H, W = self.height, self.width
        for index, blk in enumerate(self.blocks):
            s = time.time()
            x = blk(x)
            print("blk1-{}:".format(index), x.shape)
            print(time.time() - s)
        for index, block in enumerate(self.upsample_blocks):
            s = time.time()
            x, H, W = upscale(x, H, W)
            print("blk2-{} upscale:".format(index), x.shape)
            print(time.time() - s)
            for b in block:
                s = time.time()
                x = b(x)
                print("blk2-{}:".format(index), x.shape)
                print(time.time() - s)
        output = x.view(-1)
        x = x.permute(0, 2, 1)
        print(x.shape)
        x = x.view(-1, self.embed_dim//16, H, W)
        print(x.shape)
        output = self.deconv(x)

        return output


def upscale(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0, 2, 1)
    return x, H, W
