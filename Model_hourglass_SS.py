from Component import Rse_block
from Component import Rse_blockT
from Component import Short_cut_block
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from numpy import math
import numpy as np
import matplotlib.pyplot as plt


class Extractor(nn.Module):
    def __init__(self, in_ch=3):
        super(Extractor, self).__init__()
        self.E1 = Rse_block(in_ch, 32, pool=False)
        self.E2 = Rse_block(32, 64)
        self.E3 = Rse_block(64, 256)
        self.E4 = Rse_block(256, 512, last=True)

    def forward(self, input):
        e1 = self.E1(input)
        e2 = self.E2(e1)
        e3 = self.E3(e2)
        e4 = self.E4(e3)
        return e1, e2, e3, e4


class Decoder(nn.Module):
    def __init__(self, out_ch=3):
        super(Decoder, self).__init__()
        self.D3 = Rse_blockT(512 + 256, 256)
        self.D2 = Rse_blockT(256 + 128, 128)
        self.D1 = Rse_blockT(128 + 64, 64, last=True)
        self.conv_final = nn.Conv2d(64, out_ch, 1, 1)
        self.short1 = Short_cut_block(32, 64)
        self.short2 = Short_cut_block(64, 128)
        self.short3 = Short_cut_block(256, 256)

    def forward(self, e1, e2, e3, e4):
        s1 = self.short1(e1)
        s2 = self.short2(e2)
        s3 = self.short3(e3)
        d3 = self.D3(torch.cat((e4, s3), dim=1))
        d2 = self.D2(torch.cat((d3, s2), dim=1))
        d1 = self.D1(torch.cat((d2, s1), dim=1))
        result = self.conv_final(d1)
        result = torch.nn.functional.normalize(result, 2, 1)
        return result


class Sorter(nn.Module):
    def __init__(self, out_ch=3, num_perm=2, bn=True):
        super(Sorter, self).__init__()
        self.E1 = Rse_block(512, 256, bn=bn)
        self.E2 = Rse_block(256, 128, bn=bn)
        self.E3 = Rse_block(128, 32, bn=bn)
        self.f1 = nn.Linear(32 * 5 * 5, 512)
        self.f2 = nn.Linear(512, 256)
        self.f3 = nn.Linear(256, num_perm)
        # self.E4 = Rse_block(64, 32, bn=bn)
        # self.E5 = nn.Conv2d(32, num_perm, kernel_size=[2,2], stride=1, padding=0)
        self.num_perm = num_perm

    def forward(self, e1, e2, e3, e4):
        e1 = self.E1(e4)
        e2 = self.E2(e1)
        e3 = self.E3(e2)
        e = self.f1(e3.view(-1, 32 * 5 * 5))
        e = torch.nn.functional.relu(e)
        e = self.f2(e)
        e = torch.nn.functional.relu(e)
        e = self.f3(e)

        # e4 = self.E4(e3)
        # e = self.E5(e4).view(-1,self.num_perm)
        e = torch.nn.functional.softmax(e, dim=1)
        return e


class Regressor(nn.Module):
    def __init__(self, out_ch=3):
        super(Regressor, self).__init__()
        self.D3 = Rse_blockT(512 + 256, 256)
        self.D2 = Rse_blockT(256 + 128, 128)
        self.D1 = Rse_blockT(128 + 64, 64, last=True)
        self.conv_final = nn.Conv2d(64, out_ch, 1, 1)
        self.short1 = Short_cut_block(32, 64)
        self.short2 = Short_cut_block(64, 128)
        self.short3 = Short_cut_block(256, 256)

    def forward(self, e1, e2, e3, e4):
        s1 = self.short1(e1)
        s2 = self.short2(e2)
        s3 = self.short3(e3)
        d3 = self.D3(torch.cat((e4, s3), dim=1))
        d2 = self.D2(torch.cat((d3, s2), dim=1))
        d1 = self.D1(torch.cat((d2, s1), dim=1))
        result = self.conv_final(d1)
        return result


class SUNET(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, ss=True, num_puzzle=9, bench=False):
        super(SUNET, self).__init__()
        self.in_ch = in_ch
        self.extractor = Extractor(in_ch)
        self.regressor = Regressor(out_ch)
        self.num_puzzle = num_puzzle
        self.decoder = Sorter(in_ch, num_perm=min(math.factorial(self.num_puzzle), 50))
        if ss:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        self.opt_all = optim.Adam(self.parameters(), lr=0.001)
        if ss:
            self.opt_ext = optim.Adam(self.extractor.parameters(), lr=0.001)
            self.opt_dec = optim.Adam(self.decoder.parameters(), lr=0.001)
            self.scheduler_dec = StepLR(self.opt_dec, step_size=1, gamma=0.5)
            self.scheduler_ext = StepLR(self.opt_ext, step_size=1, gamma=0.5)
        elif bench:
            self.opt_ext = optim.Adam(self.extractor.parameters(), lr=0.0001)
            self.opt_reg = optim.Adam(self.regressor.parameters(), lr=0.001)
            self.scheduler_ext = StepLR(self.opt_ext, step_size=1, gamma=0.5)
            self.scheduler_reg = StepLR(self.opt_reg, step_size=1, gamma=0.5)
        else:
            self.opt_ext = optim.Adam(self.extractor.parameters(), lr=0.00001)
            self.opt_reg = optim.Adam(self.regressor.parameters(), lr=0.001)
            self.scheduler_ext = StepLR(self.opt_ext, step_size=1, gamma=0.5)
            self.scheduler_reg = StepLR(self.opt_reg, step_size=1, gamma=0.5)
        self.out_ch = out_ch
        self.imshown = False

    def forward(self, data, ss=False):
        self.input = data[0].cuda()
        if ss:
            self.rec_label = data[1].cuda()
        else:
            self.label = data[1].cuda()
        e1, e2, e3, e4 = self.extractor(self.input)
        if ss:
            self.recon = self.decoder(e1, e2, e3, e4)
        else:
            self.result = self.regressor(e1, e2, e3, e4)

    def show(self, rec=True):
        # plt.ion()
        if rec:
            im2show = self.recon
            im2show = torch.split(im2show.cpu() / 2 + 0.5, 1, dim=0)
            im2show = torch.cat(im2show, dim=2)

            # plt.imshow(im2show.squeeze().permute(1,2,0).detach().cpu())
            # plt.show()

            if not self.imshown:
                self.fig, ax = plt.subplots(1, 1)
                self.im = ax.imshow(im2show.squeeze().permute(1, 2, 0).detach().cpu())
                self.imshown = True
            else:
                self.im.set_data(im2show.squeeze().permute(1, 2, 0).detach().cpu())
                self.fig.canvas.draw_idle()
            plt.pause(0.01)
        else:
            im2show = self.result[0, :, :, :]
            plt.imshow(im2show.permute(0, 2, 3, 1).cpu())
            plt.show()

    def accuracy(self):
        pred = torch.argmax(self.recon, dim=1)
        accuracy = pred == self.rec_label
        return torch.mean(accuracy.float())

    def update(self, ss=False, see=False):
        if ss:
            # mask=torch.sum(torch.abs(self.rec_label), dim=1)==0
            # mask=mask.unsqueeze(1).repeat(1,3,1,1)
            # self.Loss = self.criterion(self.recon[mask], self.rec_label[mask].detach())

            pre = self.recon
            tar = self.rec_label
            self.Loss = self.criterion(pre, tar)

            # self.opt_ext.zero_grad()
            self.opt_ext.zero_grad()
            self.opt_dec.zero_grad()
            self.Loss.backward()
            self.opt_ext.step()
            self.opt_dec.step()
        else:
            self.Loss = self.criterion(self.result, self.label.detach())
            self.opt_ext.zero_grad()
            self.opt_reg.zero_grad()
            self.Loss.backward()
            self.opt_ext.step()
            self.opt_reg.step()

    def LR_decay(self, ss=False):
        if ss:
            self.scheduler_ext.step()
            self.scheduler_dec.step()
        else:
            self.scheduler_reg.step

    def test(self, input, show=False):
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
        torch.save(self.cpu().state_dict(), save_path)
        self.cuda()

    def load_net(self, save_path):
        self.load_state_dict(torch.load(save_path))
