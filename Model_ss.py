from Component import Rse_block
from Component import Rse_blockT
from Component import Short_cut_block
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import math
import numpy as np



class Extractor(nn.Module):
    def __init__(self, in_ch=3):
        super(Extractor, self).__init__()
        self.E1 = Rse_block(in_ch, 32, pool=False)
        self.E2 = Rse_block(64, 64)
        self.E3 = Rse_block(128, 256)
        self.E4 = Rse_block(256, 512, last=True)

class SUNET(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super(SUNET, self).__init__()
        self.extractor=Extractor()
        self.in_ch = in_ch
        self.E1_iso = Rse_block(1, 32, pool=False)
        self.E2_iso = Rse_block(32, 64)
        self.D3 = Rse_blockT(512 + 256, 256)
        self.D2 = Rse_blockT(256 + 128, 128)
        self.D1 = Rse_blockT(128 + 64, 64, last=True)
        self.comp1 = nn.Conv2d(128, 64, 1)
        self.comp2 = nn.Conv2d(256, 128, 1)
        self.comp3 = nn.Conv2d(512, 256, 1)
        self.conv_final = nn.Conv2d(64, out_ch, 1, 1)
        self.short1 = Short_cut_block(64, 64)
        self.short2 = Short_cut_block(128, 128)
        self.short3 = Short_cut_block(256, 256)

        self.criterion = nn.MSELoss()
        self.opt_all = optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = StepLR(self.opt_all, step_size=1, gamma=0.5)
        self.out_ch = out_ch

    def forward(self, data):
        self.input = data[0].cuda()
        self.label = data[1].cuda()
        e11 = self.E1_iso(self.input[:, 0, :, :].unsqueeze(1))
        e12 = self.E1_iso(self.input[:, 1, :, :].unsqueeze(1))
        e13 = self.E1_iso(self.input[:, 2, :, :].unsqueeze(1))
        e10 = torch.max(e11, e12)
        e10 = torch.max(e10, e13)
        e1 = torch.cat((self.E1(self.input),e10),dim=1)

        s1 = self.short1(e1)

        e21 = self.E2_iso(e11)
        e22 = self.E2_iso(e12)
        e23 = self.E2_iso(e13)
        e20 = torch.max(e21, e22)
        e20 = torch.max(e20, e23)
        e2 = torch.cat((self.E2(e1), e20), dim=1)

        s2 = self.short2(e2)
        e3 = self.E3(e2)

        s3 = self.short3(e3)
        e4 = self.E4(e3)
        d3 = self.D3(torch.cat((e4, s3), dim=1))
        d2 = self.D2(torch.cat((d3, s2), dim=1))
        d1 = self.D1(torch.cat((d2, s1), dim=1))
        self.result = self.conv_final(d1)

    def update(self):
        self.Loss = self.criterion(self.result, self.label.detach())
        self.opt_all.zero_grad()
        self.Loss.backward()
        self.opt_all.step()

    def update_ss(self):
        self.Loss = self.criterion(self.result, self.label.detach())
        self.opt_all.zero_grad()
        self.Loss.backward()
        self.opt_all.step()

    def LR_decay(self):
        self.scheduler.step()

    def test(self, input):
        self.eval()
        self.forward(input)
        self.train()
        gt_array = self.label[:, 0, :, :].view(-1, 160 * 160)
        pre_array = self.result[:, 0, :, :].view(-1, 160 * 160)
        idp = torch.argmax(pre_array, dim=1).cpu().float()
        idg = torch.argmax(gt_array, dim=1).cpu().float()
        xp = idp % 160
        yp = (idp - xp) / 160
        xg = idg % 160
        yg = (idg - xg) / 160
        e = torch.sqrt((xp - xg) * (xp - xg) + (yp - yg) * (yp - yg))
        self.error = torch.mean(e)
        # if self.out_ch == 2:
        gt_array = self.label[:, 1, :, :].view(-1, 160 * 160)
        pre_array = self.result[:, 1, :, :].view(-1, 160 * 160)
        idp = torch.argmax(pre_array, dim=1).cpu().float()
        idg = torch.argmax(gt_array, dim=1).cpu().float()
        xp = idp % 160
        yp = (idp - xp) / 160
        xg = idg % 160
        yg = (idg - xg) / 160
        e = torch.sqrt((xp - xg) * (xp - xg) + (yp - yg) * (yp - yg))
        self.error = self.error * 0.5 + torch.mean(e) * 0.5

    def save_net(self, save_path):
        torch.save(self.cpu().state_dict(), save_path + '/net_ss1.path')
        self.cuda()
