import torch.nn as nn
import torch
import math
import numpy as np

class Res_unit(nn.Module):
    def __init__(self, in_ch=16, out_ch=16, kernel=3,pad=1, pool=True, bn=False, IN=False, DR=False, LN=False):
        super(Res_unit, self).__init__()
        layers=[]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        layers.append(nn.Conv2d(in_ch,out_ch,kernel,1,pad))
        layers.append(nn.ReLU())

        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if LN:
            layers.append(nn.GroupNorm(out_ch,out_ch))
        if IN:
            layers.append(nn.InstanceNorm2d(out_ch))
        if DR:
            layers.append(nn.Dropout(0.2))
        self.layers = nn.Sequential(*layers)
    def forward(self, input):
        x = self.layers(input)
        return x

class Rse_block(nn.Module):
    def __init__(self, in_ch=16, out_ch=16, pool=True, bn=False, IN=False, DR=False, last=False, single=False):
        super(Rse_block, self).__init__()
        self.d1 = Res_unit(in_ch=in_ch,out_ch=out_ch, bn=bn, IN=IN, DR=DR, pool=pool)
        self.single=single
        if not single:
            self.d2 = Res_unit(in_ch=out_ch,out_ch=out_ch, bn=bn, IN=IN, DR=DR, pool=False)
        self.last = False
        self.DR= nn.Dropout(p=0.5)
        if last:
            self.last = True
            self.dt=Res_unitT(in_ch=out_ch,out_ch=out_ch, kernel=2,stride=2,pad=0)


    def forward(self, input):
        if self.last:
            input=self.DR(input)


        x = self.d1(input)
        if not self.single:
            x = self.d2(x)
        if self.last:
            x=self.dt(x)
            x = self.DR(x)
        return x

class Res_unitT(nn.Module):
    def __init__(self, in_ch=16, out_ch=16, kernel=3,stride=1, pad=1, bn=False, IN=False, DR=False, LN=True):
        super(Res_unitT, self).__init__()
        layers=[]
        layers.append(nn.ConvTranspose2d(in_ch,out_ch,kernel,stride, pad))
        layers.append(nn.ReLU())

        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if LN:
            layers.append(nn.GroupNorm(out_ch,out_ch))
        if IN:
            layers.append(nn.InstanceNorm2d(out_ch))
        if DR:
            layers.append(nn.Dropout(0.2))
        self.layers = nn.Sequential(*layers)
    def forward(self, input):
        x = self.layers(input)
        return x

class Rse_blockT(nn.Module):
    def __init__(self, in_ch=16, out_ch=16, bn=False, IN=False, DR=False, last=False):
        super(Rse_blockT, self).__init__()
        if not last:
            self.d1 = Res_unitT(in_ch=in_ch,out_ch=out_ch,kernel=2,stride=2,pad=0, bn=bn, IN=IN, DR=DR)
        else:
            self.d1 = Res_unitT(in_ch=in_ch,out_ch=out_ch, bn=bn, IN=IN, DR=DR)
        self.d2 = Res_unitT(in_ch=out_ch,out_ch=out_ch, bn=bn, IN=IN, DR=DR)

    def forward(self, input):
        x = self.d1(input)
        x = self.d2(x)
        return x


class Short_cut_unit(nn.Module):
    def __init__(self, in_ch=16, out_ch=16, kernel=3,pad=1, bn=False, IN=False, DR=False, LN=True):
        super(Short_cut_unit, self).__init__()
        layers=[]
        layers.append(nn.Conv2d(in_ch,out_ch,kernel,1,pad))
        layers.append(nn.ReLU())
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if LN:
            layers.append(nn.GroupNorm(out_ch,out_ch))
        if IN:
            layers.append(nn.InstanceNorm2d(out_ch))
        if DR:
            layers.append(nn.Dropout(0.2))
        self.layers = nn.Sequential(*layers)
    def forward(self, input):
        x = self.layers(input)
        return x

class Short_cut_block(nn.Module):
    def __init__(self, in_ch=16, out_ch=16, bn=False, IN=False, DR=False):
        super(Short_cut_block, self).__init__()
        self.d1 = Short_cut_unit(kernel=1, pad=0, in_ch=in_ch,out_ch=out_ch, bn=bn, IN=IN, DR=DR)
        self.d2 = Short_cut_unit(kernel=3, in_ch=in_ch,out_ch=out_ch, bn=bn, IN=IN, DR=DR)
    def forward(self, input):
        x1 = self.d1(input)
        x2 = self.d2(input)
        return x1+x2








