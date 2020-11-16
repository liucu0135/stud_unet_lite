from Component import Rse_block
from Component import Rse_blockT
from Component import Short_cut_block
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import math
import numpy as np
from models import model_utils


class Attention_layer(nn.Module):
    def __init__(self, ch_in, batch=False, factor=1, bias=True, extra=False, shurink=0):
        super(Attention_layer, self).__init__()
        self.atten_factor = 1
        self.extra = extra
        if shurink:
            self.shurink = shurink
            ch_out=shurink
        else:
            self.shurink = ch_in
            ch_out=ch_in
        k = [1, 1, 1]
        s = [1, 1, 1]
        p = [0, 0, 0]
        self.atten_k = nn.Conv3d(ch_in, self.shurink, kernel_size=k, stride=s, padding=p, bias=bias)
        self.atten_q = nn.Conv3d(ch_in, self.shurink, kernel_size=k, stride=s, padding=p, bias=bias)
        self.atten_v = nn.Conv3d(ch_in, self.shurink, kernel_size=k, stride=s, padding=p, bias=bias)
        self.atten = nn.Conv3d(self.shurink, ch_out, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x, visual=False):
        shape = x.shape
        if self.extra:
            extra = nn.functional.avg_pool3d(x, kernel_size=[shape[2], 1, 1], stride=[1, 1, 1], padding=[0, 0, 0])
            x = torch.cat([extra, x], dim=2)
            shape = x.shape
        atten_k = self.atten_k(x).permute(0, 3, 4, 1, 2).contiguous().view(shape[0] * shape[3] * shape[4]//1, self.shurink,
                                                                           shape[2])
        atten_q = self.atten_q(x).permute(0, 3, 4, 1, 2).contiguous().view(shape[0] * shape[3] * shape[4]//1, self.shurink,
                                                                           shape[2])
        atten_v = self.atten_v(x).permute(0, 3, 4, 1, 2).contiguous().view(shape[0] * shape[3] * shape[4]//1, self.shurink,
                                                                           shape[2])
        a = nn.functional.softmax(torch.bmm(atten_k.transpose(1, 2), atten_q) / math.sqrt(self.shurink), dim=1)
        a = torch.bmm(atten_v, a)
        a = self.atten(a.view(shape[0], shape[3]//1, shape[4]//1, self.shurink, shape[2]).permute(0, 3, 4, 1, 2))
        # x = self.atten_factor * a
        x = x + self.atten_factor * a
        return x


class FeatExtractor(nn.Module):
    def __init__(self, ch_in=32, batchNorm=False, base_size=64):
        super(FeatExtractor, self).__init__()
        self.conv1 = model_utils.conv3d(batchNorm, 1, 16, k=[1, 1, 1], stride=1, pad=[0, 0, 0])
        self.at1 = Attention_layer(16)
        self.bn1 = nn.GroupNorm(16, 16)

    def forward(self, x):
        x = x.unsqueeze(1)
        shape = x.shape
        out = self.conv1(x)
        out = self.at1(out)
        out = self.bn1(out)
        # out0 = nn.functional.max_pool3d(out0, kernel_size=[shape[2], 1, 1], stride=[shape[2], 1, 1],
        #                                 padding=0).squeeze()
        out=out.view(-1,16*16,160,160)
        return out


class SUNET(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, atten=False):
        super(SUNET, self).__init__()
        base_size = 64
        self.extractor = FeatExtractor(ch_in=1, batchNorm=True, base_size=base_size*2)
        self.in_ch = in_ch
        self.E1 = Rse_block(16*16, base_size,pool=False)
        self.E2 = Rse_block(base_size, base_size*2)
        self.E3 = Rse_block(base_size * 2, base_size * 4)
        self.E4 = Rse_block(base_size * 4, base_size * 8, last=True)
        self.D3 = Rse_blockT(base_size * 8 + base_size * 4, base_size * 4)
        self.D2 = Rse_blockT(base_size * 4 + base_size * 2, base_size * 2)
        self.D1 = Rse_blockT(base_size * 2 + base_size, base_size, last=True)
        self.comp1 = nn.Conv2d(base_size * 2, base_size, 1)
        self.comp2 = nn.Conv2d(base_size * 4, base_size * 2, 1)
        self.comp3 = nn.Conv2d(base_size * 8, base_size * 4, 1)
        self.conv_final = nn.Conv2d(base_size, out_ch, 1, 1)
        self.short1 = Short_cut_block(base_size, base_size)
        self.short2 = Short_cut_block(base_size * 2, base_size * 2)
        self.short3 = Short_cut_block(base_size * 4, base_size * 4)

        self.criterion = nn.MSELoss()
        self.opt_all = optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = StepLR(self.opt_all, step_size=1, gamma=0.5)
        self.out_ch = out_ch

    def forward(self, data):
        input = data[0].cuda()
        self.label = data[1].cuda()
        e1= self.extractor(input)
        e1 = self.E1(e1)
        e2 = self.E2(e1)
        e3 = self.E3(e2)
        s1 = self.short1(e1)
        s2 = self.short2(e2)
        s3 = self.short3(e3)
        e4 = self.E4(e3)
        d3 = self.D3(torch.cat((e4, s3), dim=1))
        d2 = self.D2(torch.cat((d3, s2), dim=1))
        d1 = self.D1(torch.cat((d2, s1), dim=1))
        self.result = self.conv_final(d1)
        return self.result

    def update(self):
        self.Loss = self.criterion(self.result, self.label.detach())
        self.opt_all.zero_grad()
        self.Loss.backward()
        self.opt_all.step()
        self.result = 0
        self.label = 0
        return self.Loss.detach().cpu()

    def LR_decay(self):
        self.scheduler.step()

    def test(self, input):
        self.eval()
        self.forward(input)
        self.train()
        size = 160
        gt_array = self.label[:, 0, :, :].view(-1, size * size)
        pre_array = self.result[:, 0, :, :].view(-1, size * size)
        idp = torch.argmax(pre_array, dim=1).cpu().float()
        idg = torch.argmax(gt_array, dim=1).cpu().float()
        xp = idp % size
        yp = (idp - xp) / size
        xg = idg % size
        yg = (idg - xg) / size
        e = torch.sqrt((xp - xg) * (xp - xg) + (yp - yg) * (yp - yg))
        self.error = torch.mean(e)
        # if self.out_ch == 2:
        gt_array = self.label[:, 1, :, :].view(-1, size * size)
        pre_array = self.result[:, 1, :, :].view(-1, size * size)
        idp = torch.argmax(pre_array, dim=1).cpu().float()
        idg = torch.argmax(gt_array, dim=1).cpu().float()
        xp = idp % size
        yp = (idp - xp) / size
        xg = idg % size
        yg = (idg - xg) / size
        e = torch.sqrt((xp - xg) * (xp - xg) + (yp - yg) * (yp - yg))
        self.error = self.error * 0.5 + torch.mean(e) * 0.5

    def save_net(self, save_path):
        torch.save(self.cpu().state_dict(), save_path + '/net_ss1.path')
        self.cuda()
