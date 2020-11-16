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
from torchvision import utils as vutils

class Extractor(nn.Module):
    def __init__(self, in_ch=1):
        super(Extractor, self).__init__()
        self.E1 = Rse_block(in_ch, 16, pool=False)
        self.E2 = Rse_block(16, 32)
        self.E3 = Rse_block(32, 128)
        self.E4 = Rse_block(128, 256, last=True)

    def forward(self, input, ri=False):
        e1 = self.E1(input)
        e2 = self.E2(e1)
        e3 = self.E3(e2)
        e4 = self.E4(e3)

        return e1, e2, e3, e4

class Regressor(nn.Module):
    def __init__(self, out_ch=3, bn=False):
        super(Regressor, self).__init__()
        # layers = [Rse_block(512+256, 128, bn=True, single=True),
        #           Rse_block(128, 64, bn=True, single=True),
        #           Rse_block(64, 8, bn=True, single=True)]
        layers = [Rse_block(512+256, 128, bn=bn, single=True),
                  # Rse_block(128, 128, bn=True, single=True),
                  # Rse_block(128, 128, bn=True, single=True),
                  Rse_block(128, 64, bn=bn, single=True),
                  Rse_block(64, 8, bn=bn, single=True)]
        self.f1 = nn.Linear(8 * 5 * 5, 4)
        # self.f2 = nn.Linear(64, 2)
        self.clasifier = nn.Sequential(*layers)
        self.num_perm = 2

    def forward(self, e1, e2, e3, e4):
        e = self.clasifier(e4)
        e = self.f1(e.view(-1, 8 * 5 * 5))
        # e = torch.nn.functional.softmax(e, dim=1)
        return e

# class Decoder(nn.Module):
#     def __init__(self, out_ch=1):
#         super(Decoder, self).__init__()
#         self.D3 = Rse_blockT(256 + 128, 128)
#         self.D2 = Rse_blockT(128 + 32, 64)
#         self.D1 = Rse_blockT(64 + 16, 32, last=True)
#         self.conv_final = nn.Conv2d(32, out_ch, 1, 1)
#         self.short1 = Short_cut_block(16, 16)
#         self.short2 = Short_cut_block(32, 32)
#         self.short3 = Short_cut_block(128, 128)
#
#     def forward(self, e1, e2, e3, e4):
#         s1 = self.short1(e1)
#         s2 = self.short2(e2)
#         s3 = self.short3(e3)
#         d3 = self.D3(torch.cat((e4, s3), dim=1))
#         d2 = self.D2(torch.cat((d3, s2), dim=1))
#         d1 = self.D1(torch.cat((d2, s1), dim=1))
#         result = self.conv_final(d1)
#         result = torch.nn.functional.tanh(result)
#         return result
class Decoder(nn.Module):
    def __init__(self, out_ch=1):
        super(Decoder, self).__init__()
        self.D3 = Rse_blockT(256 + 128, 128)
        self.D2 = Rse_blockT(128 + 32, 64)
        self.D1 = Rse_blockT(64 + 16, 32, last=True)
        self.conv_final = nn.Conv2d(32, out_ch, 1, 1)
        self.short1 = Short_cut_block(16, 16)
        self.short2 = Short_cut_block(32, 32)
        self.short3 = Short_cut_block(128, 128)

    def forward(self, e1, e2, e3, e4):
        d3 = self.D3(e4, dim=1)
        d2 = self.D2(d3, dim=1)
        d1 = self.D1(d2, dim=1)
        result = self.conv_final(d1)
        result = torch.nn.functional.tanh(result)
        return result


# class Regressor(nn.Module):
#     def __init__(self, out_ch=3, bn=False):
#         super(Regressor, self).__init__()
#         self.D3 = Rse_blockT(256*3 + 256, 128*3)
#         self.D2 = Rse_blockT(128*3 + 128, 64*3)
#         self.D1 = Rse_blockT(64*3 + 64, 32*3, last=True)
#         self.conv_final = nn.Conv2d(32*3, out_ch, 1, 1)
#         self.short1 = Short_cut_block(16*3, 64)
#         self.short2 = Short_cut_block(32*3, 128)
#         self.short3 = Short_cut_block(128*3, 256)
#
#     def forward(self, e1, e2, e3, e4):
#         s1 = self.short1(e1)
#         s2 = self.short2(e2)
#         s3 = self.short3(e3)
#         d3 = self.D3(torch.cat((e4, s3), dim=1))
#         d2 = self.D2(torch.cat((d3, s2), dim=1))
#         d1 = self.D1(torch.cat((d2, s1), dim=1))
#         result = self.conv_final(d1)
#         return result


class SUNET(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, ss=True, num_puzzle=9, bench=False):
        super(SUNET, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.imshown = False
        self.extractor0 = Extractor(1)
        self.extractor1 = Extractor(1)
        self.extractor2 = Extractor(1)
        if ss:
            self.decoder = Decoder(1)
            self.criterion = nn.MSELoss()
            self.opt_ext = optim.Adam(list(self.extractor0.parameters())+list(self.extractor1.parameters())+list(self.extractor2.parameters()), lr=0.001)
            self.opt_dec = optim.Adam(self.decoder.parameters(), lr=0.001)
            self.scheduler_dec = StepLR(self.opt_dec, step_size=1, gamma=0.5)
            self.scheduler_ext = StepLR(self.opt_ext, step_size=1, gamma=0.5)
        else:
            self.regressor = Regressor(out_ch)
            self.criterion = nn.MSELoss()
            p0=list(self.extractor0.E4.parameters())+list(self.extractor0.E3.parameters())
            p1=list(self.extractor1.E4.parameters())+list(self.extractor0.E3.parameters())
            p2=list(self.extractor2.E4.parameters())+list(self.extractor0.E3.parameters())
            # self.opt_ext = optim.Adam(p0+p1+p2, lr=0.001)
            self.opt_ext = optim.Adam(list(self.extractor0.parameters())+list(self.extractor1.parameters())+list(self.extractor2.parameters()), lr=0.001)
            self.opt_reg = optim.Adam(self.regressor.parameters(), lr=0.001)
            self.scheduler_ext = StepLR(self.opt_ext, step_size=1, gamma=0.5)
            self.scheduler_reg = StepLR(self.opt_reg, step_size=1, gamma=0.5)

    def forward(self, data, ss=True, g=True):
        if ss:
            input = data[0].cuda()
            input0=input[:,0,:,:].unsqueeze(1)
            input1=input[:,1,:,:].unsqueeze(1)
            input2=input[:,2,:,:].unsqueeze(1)
            self.label0 = input1
            self.label1 = input2
            self.label2 = input0
            e11, e21, e31, e41 = self.extractor0(input0)
            e12, e22, e32, e42 = self.extractor1(input1)
            e13, e23, e33, e43 = self.extractor2(input2)
            self.result0 = self.decoder(e11, e21, e31, e41)
            self.result1 = self.decoder(e12, e22, e32, e42)
            self.result2 = self.decoder(e13, e23, e33, e43)

        else:
            input = data[0].cuda()

            input0=input[:,0,:,:].unsqueeze(1)
            input1=input[:,1,:,:].unsqueeze(1)
            input2=input[:,2,:,:].unsqueeze(1)
            e11, e21, e31, e41 = self.extractor0(input0)
            e12, e22, e32, e42 = self.extractor1(input1)
            e13, e23, e33, e43 = self.extractor2(input2)
            e1=torch.cat((e11, e12, e13), dim=1)
            e2=torch.cat((e21, e22, e23), dim=1)
            e3=torch.cat((e31, e32, e33), dim=1)
            e4=torch.cat((e41, e42, e43), dim=1)
            self.result = self.regressor(e1, e2, e3, e4)


            self.label = data[1]
            gt_array = self.label[:, 0, :, :].view(-1, 160 * 160)
            idg = torch.argmax(gt_array, dim=1).float()
            xg1 = idg % 160
            yg1 = (idg - xg1) / 160
            gt_array = self.label[:, 1, :, :].view(-1, 160 * 160)
            idg = torch.argmax(gt_array, dim=1).float()
            xg2 = idg % 160
            yg2 = (idg - xg2) / 160
            self.label = torch.stack([xg1, yg1, xg2, yg2], dim=1).cuda()

    def show(self, rec=True):
        # plt.ion()
        if rec:
            im2show = self.result0[0,0,:,:]/2+0.5
            vutils.save_image(im2show, 'img_output/img_result0.png')
            im2show = self.result1[0,0,:,:]/2+0.5
            vutils.save_image(im2show, 'img_output/img_result1.png')
            im2show = self.result2[0,0,:,:]/2+0.5
            vutils.save_image(im2show, 'img_output/img_result2.png')
            im2show = self.label0[0,0,:,:]/2+0.5
            vutils.save_image(im2show, 'img_output/img_label0.png')
            im2show = self.label1[0,0,:,:]/2+0.5
            vutils.save_image(im2show, 'img_output/img_label1.png')
            im2show = self.label2[0,0,:,:]/2+0.5
            vutils.save_image(im2show, 'img_output/img_label2.png')

        #     if not self.imshown:
        #         self.fig, ax = plt.subplots(3, 2)
        #         self.im = ax.imshow(im2show.detach().cpu())
        #         self.imshown = True
        #     else:
        #         self.im.set_data(im2show.detach().cpu())
        #         self.fig.canvas.draw_idle()
        #     plt.pause(0.01)
        # else:
        #     im2show = self.result[0, :, :, :]
        #     plt.imshow(im2show.permute(0, 2, 3, 1).cpu())
        #     plt.show()


    def cal_loss_ss(self, ss_only=False):
        self.Loss0 = self.criterion(self.result0, self.label0)
        self.Loss1 = self.criterion(self.result1, self.label1)
        self.Loss2 = self.criterion(self.result2, self.label2)
        self.Loss=self.Loss0+self.Loss1+self.Loss2

    def cal_loss(self):
        self.Loss = self.criterion(self.result, self.label)

    def update(self, ss=False, ext_fix=False):
        if ss:
            self.cal_loss_ss()
            self.opt_ext.zero_grad()
            self.opt_dec.zero_grad()
            self.Loss.backward()
            self.opt_ext.step()
            self.opt_dec.step()
        else:
            self.cal_loss()
            if not ext_fix:
                self.opt_ext.zero_grad()
            self.opt_reg.zero_grad()
            self.Loss.backward()
            if not ext_fix:
                self.opt_ext.step()
            self.opt_reg.step()


    def LR_decay(self, ss=False):
        if ss:
            self.scheduler_ext.step()
            self.scheduler_dec.step()
        else:
            self.scheduler_reg.step()



    def test(self, input):
        self.eval()
        self.forward(input, ss=False)
        error = (self.result - self.label).detach().cpu()
        self.error = torch.mean(
            torch.sqrt(error[:, 1] ** 2 + error[:, 0] ** 2) + torch.sqrt(error[:, 3] ** 2 + error[:, 2] ** 2)) / 2

        self.train()


    # def test(self, input, show=False):
    #     self.eval()
    #     self.forward(input, ss=False)
    #     self.train()
    #     gt_array = self.label[:, 0, :, :].view(-1, 160 * 160)
    #     pre_array = self.result[:, 0, :, :].view(-1, 160 * 160)
    #     idp = torch.argmax(pre_array, dim=1).cpu().float()
    #     idg = torch.argmax(gt_array, dim=1).cpu().float()
    #     xp = idp % 160
    #     yp = (idp - xp) / 160
    #     xg = idg % 160
    #     yg = (idg - xg) / 160
    #     e = torch.sqrt((xp - xg) * (xp - xg) + (yp - yg) * (yp - yg))
    #     self.error = torch.mean(e)
    #     # if self.out_ch == 2:
    #     gt_array = self.label[:, 1, :, :].view(-1, 160 * 160)
    #     pre_array = self.result[:, 1, :, :].view(-1, 160 * 160)
    #     idp = torch.argmax(pre_array, dim=1).cpu().float()
    #     idg = torch.argmax(gt_array, dim=1).cpu().float()
    #     xp = idp % 160
    #     yp = (idp - xp) / 160
    #     xg = idg % 160
    #     yg = (idg - xg) / 160
    #     e = torch.sqrt((xp - xg) * (xp - xg) + (yp - yg) * (yp - yg))
    #     self.error = self.error * 0.5 + torch.mean(e) * 0.5

    def save_net(self, save_path):
        torch.save(self.cpu().state_dict(), save_path)
        self.cuda()

    def load_net(self, save_path, ext_only=False):
        if ext_only:
            state_dict = torch.load(save_path)
            s0 = {k: state_dict[k] for k in state_dict if 'extractor0.E' in k}
            s1 = {k: state_dict[k] for k in state_dict if 'extractor1.E' in k}
            s2 = {k: state_dict[k] for k in state_dict if 'extractor2.E' in k}
            self.extractor0.load_state_dict(s0, strict=False)
            self.extractor1.load_state_dict(s1, strict=False)
            self.extractor2.load_state_dict(s2, strict=False)
        else:
            self.load_state_dict(torch.load(save_path))
