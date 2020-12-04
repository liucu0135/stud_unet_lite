import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from numpy import math
import numpy as np
from torch.optim.lr_scheduler import StepLR

from Component import Rse_block
from Component import Rse_blockT
from Component import Short_cut_block


class Compensator(nn.Module):
    def __init__(self, in_ch=3):
        super(Compensator, self).__init__()
        self.E1 = Rse_block(in_ch, 32, pool=False)
        self.E2 = Rse_block(32, 64)
        self.E3 = Rse_block(64, 256)
        # self.E4 = Rse_block(256, 512, last=True)

    def forward(self, input):
        e1 = self.E1(input)
        e2 = self.E2(e1)
        e3 = self.E3(e2)
        # e4 = self.E4(e3)
        return e1, e2, e3


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




class Classifier(nn.Module):
    def __init__(self, out_ch=3, bn=False, single=False, para_reduce=1):
        super(Classifier, self).__init__()
        layers = [Rse_block(128//para_reduce, 128, bn=bn, single=single,DR=False),
                  Rse_block(128, 64, pool=False, bn=bn, single=single),
                  # nn.Dropout2d(0.5),
                  Rse_block(64, 32, bn=bn, single=single,DR=False)]
        self.f1 = nn.Linear(8 * 10 * 10, 64)
        self.f2 = nn.Linear(64, 5)
        self.clasifier = nn.Sequential(*layers)

    def forward(self, e1, e2, e3, e4):
        # e = self.E2(e1)
        # e = torch.cat((e, e2), dim=1)
        # e = self.E3(e)
        # e = torch.cat((e, e3), dim=1)
        e = self.clasifier(e3)
        # e = torch.nn.functional.dropout(e,0.2)
        e = self.f1(e.view(-1, 8 * 10 * 10))
        e = torch.nn.functional.relu(e)
        e = self.f2(e)
        e = torch.nn.functional.softmax(e, dim=1)
        return e





class Regressor(nn.Module):
    def __init__(self, out_ch=3, bn=False, para_reduce=1):
        super(Regressor, self).__init__()
        self.D3 = Rse_blockT(512//para_reduce+256//para_reduce, 256//para_reduce, bn=bn)
        self.D2 = Rse_blockT(256//para_reduce+64//para_reduce, 128//para_reduce, bn=bn)
        self.D1 = Rse_blockT(128//para_reduce+32//para_reduce, 64//para_reduce, bn=bn, last=True)
        self.conv_final = nn.Conv2d(64//para_reduce, out_ch, 1, 1)
        # self.short1 = Short_cut_block(32//para_reduce, 64//para_reduce, bn=bn)
        # self.short2 = Short_cut_block(64//para_reduce, 128//para_reduce, bn=bn)
        # self.short3 = Short_cut_block(256//para_reduce, 64//para_reduce, bn=bn)

    def forward(self, e1, e2, e3, e4):
        # s1 = self.short1(e1)
        # s2 = self.short2(e2)
        # s3 = self.short3(e3)
        s1=e1
        s2=e2
        s3=e3
        d3 = self.D3(torch.cat((e4, s3), dim=1))
        d2 = self.D2(torch.cat((d3, s2), dim=1))
        # d2 = self.D2(d3)
        d1 = self.D1(torch.cat((d2, s1), dim=1))
        # d1 = self.D1(d2)
        result = self.conv_final(d1)
        return result

class Regressor_ff(nn.Module):
    def __init__(self, out_ch=3, bn=False, single=True, para_reduce=1):
        super(Regressor_ff, self).__init__()
        # layers = [Rse_block(512+256, 128, bn=True, single=True),
        #           Rse_block(128, 64, bn=True, single=True),
        #           Rse_block(64, 8, bn=True, single=True)]
        layers = [Rse_block(128//para_reduce, 128, bn=bn, single=single,DR=False),
                  Rse_block(128, 64, pool=False, bn=bn, single=single),
                  # nn.Dropout2d(0.5),
                  Rse_block(64, 32, bn=bn, single=single,DR=False)]
        self.para_reduce=para_reduce
        self.f1 = nn.Linear(32 * 20 * 20, 4)
        self.f2 = nn.Linear(256, 4)
        self.clasifier = nn.Sequential(*layers)


    def forward(self, e):
        # e4 = e1
        # e4 = torch.cat((e4, e3), dim=1)
        e = self.clasifier(e)
        # e = torch.nn.functional.relu(e)
        # e = torch.nn.functional.dropout(e, 0.2)
        e = self.f1(e.view(-1, 32 * 20 * 20))
        # e = torch.nn.functional.dropout(e,0.2)
        # e = torch.nn.functional.relu(e)
        # e = self.f2(e)
        # e = torch.nn.functional.relu(e)
        # e = self.f2(e)
        # e = torch.nn.functional.softmax(e, dim=1)
        return e

class Extractor(nn.Module):
    def __init__(self, in_ch=3, bn=False, para_reduce=1):
        super(Extractor, self).__init__()
        self.E1 = Rse_block(in_ch, 32//para_reduce, pool=False, bn=bn, single=True)
        # self.E1 = Rse_block(in_ch, 64, pool=False)
        self.E2 = Rse_block(32//para_reduce, 64//para_reduce, bn=bn, single=True)
        # self.E2_ri = Rse_block(32, 64)
        self.E3 = Rse_block(64//para_reduce, 128//para_reduce, pool=False, bn=bn, single=True)
        # self.E4 = Rse_block(256//para_reduce, 512//para_reduce, last=True)

        # self.comp = Compensator(in_ch)
        # self.opt_comp = optim.Adam(self.comp.parameters(), lr=0.001)
        # self.opt_self = optim.Adam(
        #     list(self.E1.parameters()) + list(self.E2.parameters()) + list(self.E3.parameters()) + list(
        #         self.E4.parameters()), lr=0.001)

    def forward(self, input, ri=False):

        e1 = self.E1(input)
        e2 = self.E2(e1)
        e3 = self.E3(e2)
        # e4 = self.E4(e3)
        return e3

class Discriminator(nn.Module):
    def __init__(self, out_ch=3, bn=False,para_reduce=1):
        super(Discriminator, self).__init__()
        # layers = [Rse_block(512+256, 128, bn=True, single=True),
        #           Rse_block(128, 64, bn=True, single=True),
        #           Rse_block(64, 8, bn=True, single=True)]
        layers = [Rse_block((512)//para_reduce*4, 64//para_reduce, bn=bn, single=True),
                  # Rse_block(64//para_reduce, 32//para_reduce, bn=bn, single=True),
                  Rse_block(64//para_reduce, 64, bn=bn, single=True)]
        # self.E2 = Rse_block(64//para_reduce, 64//para_reduce, bn=bn)
        # self.E3 = Rse_block((256+64)//para_reduce, 64//para_reduce, bn=bn, pool=False)
        self.f1 = nn.Linear(128* 12* 12//para_reduce, 64//para_reduce)
        self.f2 = nn.Linear(64//para_reduce, 9)
        self.clasifier = nn.Sequential(*layers)
        self.num_perm = 2
        self.para_reduce=para_reduce

    def forward(self,e4):
        # e = self.clasifier(e4)
        e = self.f1(e4.view(-1, 128* 12* 12//self.para_reduce))
        e = torch.nn.functional.relu(e)
        e = self.f2(e)
        e = torch.nn.functional.softmax(e, dim=1)
        return e

class Sorter(nn.Module):
    def __init__(self, out_ch=3, num_perm=2, bn=True, para_reduce=1):
        super(Sorter, self).__init__()

        layers=[nn.Conv2d(512//para_reduce*4, 512//para_reduce,kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(512 // para_reduce),
                nn.Conv2d(512 // para_reduce , 256 // para_reduce, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(256 // para_reduce),
                nn.Conv2d(256 // para_reduce, 64 // para_reduce, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(64 // para_reduce),
                ]
        self.E=nn.Sequential(*layers)
        self.para_reduce=para_reduce
        self.f1 = nn.Linear(128*9 * 12* 12//para_reduce, 64//para_reduce)
        self.dr1 = nn.Dropout(0.5)
        self.f3 = nn.Linear(64//para_reduce, num_perm)
        self.num_perm = num_perm

    def forward(self, e):

        e = self.f1(e.view(-1,128*9 * 12* 12//self.para_reduce))
        e = torch.nn.functional.relu(e)
        e = self.f3(e)
        e = torch.nn.functional.softmax(e, dim=1)
        return e

class SUNET(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, ss=True, num_puzzle=9, scale_lr=1, multitask=False, para_reduce=1, ff=False, train_ext=True):
        super(SUNET, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.imshown = False
        self.ff=ff
        self.multi=multitask
        self.extractor = Extractor(in_ch, para_reduce=para_reduce)
        if multitask:
            self.classifier=Classifier(bn=False,para_reduce=para_reduce)
            self.opt_c = optim.Adam(self.classifier.parameters(), lr=0.001, weight_decay=0.001)
            self.scheduler_cls = StepLR(self.opt_c, step_size=1, gamma=0.5)
        if ss:
            self.disc = Discriminator(para_reduce=para_reduce, bn=False)
            self.num_puzzle = num_puzzle
            self.decoder = Sorter(in_ch, num_perm=min(math.factorial(self.num_puzzle), 50),para_reduce=para_reduce)
            self.criterion = nn.CrossEntropyLoss()
            self.opt_dics = optim.Adam(self.disc.parameters(), lr=0.001)
            self.opt_ext = optim.Adam(self.extractor.parameters(), lr=0.001)
            self.opt_dec = optim.Adam(self.decoder.parameters(), lr=0.001)
            self.scheduler_dec = StepLR(self.opt_dec, step_size=1, gamma=0.5)
            self.scheduler_ext = StepLR(self.opt_ext, step_size=1, gamma=0.5)
        else:
            if self.ff:
                self.regressor = Regressor_ff(out_ch, para_reduce=para_reduce,bn=False)
            else:
                self.regressor = Regressor(out_ch, para_reduce=para_reduce)
            self.criterion = nn.MSELoss()
            # paras1=list(self.extractor.parameters())
            paras1=list(self.extractor.E1.parameters())
            paras2=list(self.regressor.parameters())
            paras2+=list(self.extractor.E3.parameters())+list(self.extractor.E2.parameters())#+list(self.regressor.parameters())
            # self.opt_ext = optim.Adam(paras, lr=0.00001)
            if train_ext:
                # self.opt_ext = optim.Adam(paras1, lr=0.001*scale_lr*train_ext)
                self.opt_ext = optim.Adam(paras1, lr=0.001)
                self.scheduler_ext = StepLR(self.opt_ext, step_size=1, gamma=0.5)
                self.opt_reg = optim.Adam(paras2, lr=0.001, weight_decay=0.005)
            else:
                self.opt_ext = optim.Adam(self.extractor.parameters(), lr=0)
                self.scheduler_ext = StepLR(self.opt_ext, step_size=1, gamma=0.5)
                self.opt_reg = optim.Adam(self.regressor.parameters(), lr=0.001, weight_decay=0.005)

            self.scheduler_reg = StepLR(self.opt_reg, step_size=1, gamma=0.5)


    def forward(self, data, ss=True, ss_only=False, feat_only=False):
        # data 0, 1, 2:   normal map, corresponding raw image, another unlabeled raw image

        if feat_only:
            self.input_nm = data[0][-3]
            self.rec_label_nm = data[0][1].cuda()
            self.input_ri = data[1][-3]
            self.rec_label_ri = data[1][1].cuda()

            # e4_nm=[self.extractor(data_in.cuda()) for data_in in self.input_nm ]
            # e4_ri=[self.extractor(data_in.cuda()) for data_in in self.input_ri ]
            e4_nm=self.extractor(self.input_nm.cuda())
            e4_ri=self.extractor(self.input_ri.cuda())
            batchsize=self.rec_label_nm.shape[0]

            return e4_nm.view(batchsize, -1).detach().cpu(),e4_ri.view(batchsize, -1).detach().cpu()


        if ss:
            self.input_nm = data[0][-1]
            self.rec_label_nm = data[0][1].cuda()
            self.input_rip = data[2][-1]
            self.rec_label_rip = data[2][1].cuda()
            self.input_ri = data[1][-1]
            self.rec_label_ri = data[1][1].cuda()

            e4_nm=[self.extractor(data_in.cuda()) for data_in in self.input_nm ]
            e4_ri=[self.extractor(data_in.cuda()) for data_in in self.input_ri ]
            e4_rip=[self.extractor(data_in.cuda()) for data_in in self.input_rip ]

            # # mix features
            # mix_label=torch.from_numpy(np.random.randint(0,2,[len(e4_nm)]).astype(np.float)).cuda()
            #
            # e4_mix=[n*m+r*(1-m) for n,r,m in zip(e4_nm, e4_ri, mix_label)]



            self.recon_nm = self.decoder(torch.cat(e4_nm, dim=1))
            self.recon_rip = self.decoder(torch.cat(e4_rip, dim=1))
            self.recon_ri = self.decoder(torch.cat(e4_ri, dim=1))
            # if self.multi:
            #     self.c_pre=self.classifier(_, _, _, e4_nm)
            #     self.c_label = data[0][2].cuda()

            e4_nm=torch.cat(e4_nm, dim=0)
            e4_ri=torch.cat(e4_ri, dim=0)
            self.d_pre = torch.cat((self.disc(e4_nm), self.disc(e4_ri)), dim=0)
            self.d_label = torch.cat(
                (torch.ones(e4_nm.shape[0], dtype=torch.long), torch.zeros(e4_nm.shape[0], dtype=torch.long)),
                dim=0).cuda()
            self.g_label = torch.cat(
                (torch.ones(e4_nm.shape[0], dtype=torch.long), torch.ones(e4_nm.shape[0], dtype=torch.long)),
                dim=0).cuda()

        else:
            self.input = data[0].cuda()
            self.label = data[1].cuda()
            self.c_label=data[2].cuda()

            e4 = self.extractor(self.input)

            if self.multi:
                self.c_pre=self.classifier(e4)
            else:
                self.result = self.regressor(e4)

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

    def accuracy_class(self):
        if self.multi and self.c_pre.shape[0] == self.c_label.shape[0]:
            pred = torch.argmax(self.c_pre, dim=1)
            accuracy = pred == self.c_label
            a = torch.mean(accuracy.float())
            return a.detach().cpu()
        else:
            return torch.tensor(-10000.0)

    def accuracy_gan(self):
        pred = torch.argmax(self.d_pre, dim=1)
        if self.d_pre.shape[0] == self.d_label.shape[0]:
            accuracy = pred == self.d_label
            a = torch.mean(accuracy.float())
            return a.detach().cpu()
        else:
            return torch.tensor(0.0)

    def accuracy(self, domain=None):

        if domain=='ri':
            pred = torch.argmax(self.recon_ri, dim=1)
            accuracy = pred == self.rec_label_ri
            a = torch.mean(accuracy.float())
        elif domain=='rip':
            pred = torch.argmax(self.recon_rip, dim=1)
            accuracy = pred == self.rec_label_rip
            a = torch.mean(accuracy.float())
        else:
            pred = torch.argmax(self.recon_nm, dim=1)
            accuracy = pred == self.rec_label_nm
            a = torch.mean(accuracy.float())
        return a.detach().cpu()

    def cal_loss_g(self, ss_only=False):
        self.Loss_rec_nm = self.criterion(self.recon_nm, self.rec_label_nm)
        self.Loss_rec_ri = self.criterion(self.recon_ri, self.rec_label_ri)
        self.Loss_rec_rip = self.criterion(self.recon_rip, self.rec_label_rip)
        if not ss_only:
            bs=self.d_pre.shape[0]//2
            self.Loss_g = self.criterion(self.d_pre[bs:], self.g_label[bs:])
        # if self.multi:
        #     self.cal_loss_c()

    def cal_loss(self):
        if self.ff:
            self.Loss = self.criterion(self.result, self.heatmap2coord(self.label))
        else:
            self.Loss = self.criterion(self.result, self.label)

    def update(self, retain_graph=False, multi=False, reg_only=False):
        self.cal_loss()
        if self.multi:
            self.cal_loss_c()
            self.Loss+=self.Loss_c
            self.opt_c.zero_grad()

        if not reg_only:
            self.opt_ext.zero_grad()
        self.opt_reg.zero_grad()
        self.Loss.backward(retain_graph=retain_graph)
        if not reg_only:
            self.opt_ext.step()

        if self.multi:
            self.opt_c.step()
        self.opt_reg.step()

    def update_c(self):
        self.cal_loss_c()
        self.opt_ext.zero_grad()
        self.opt_c.zero_grad()
        self.Loss_c.backward()
        self.opt_ext.step()
        self.opt_c.step()

    def cal_loss_d(self):
        self.Loss_d = self.criterion(self.d_pre, self.d_label)

    def cal_loss_c(self):
        self.Loss_c = torch.nn.functional.cross_entropy(self.c_pre, self.c_label)

    def update_g(self, ss_only=False, multi=False, g_scale=1):
        self.cal_loss_g(ss_only)
        if ss_only:
            l = self.Loss_rec_rip  +self.Loss_rec_nm#+self.Loss_rec_ri
        else:
            l = self.Loss_rec_ri + self.Loss_rec_rip + self.Loss_rec_nm+ self.Loss_g*g_scale

        if multi:
            l +=self.Loss_c
        self.opt_dec.zero_grad()
        self.opt_ext.zero_grad()
        if self.multi:
            self.opt_c.zero_grad()
        l.backward(retain_graph=True)
        if self.multi:
            self.opt_c.step()
        self.opt_dec.step()
        self.opt_ext.step()

        # if not ss_only:
        #     self.update_d()
        # self.extractor.opt_self.step()
        # self.extractor.opt_comp.step()

    def update_d(self, retrain=False):
        self.cal_loss_d()
        self.opt_dics.zero_grad()
        self.Loss_d.backward(retain_graph=True)
        self.opt_dics.step()

    def LR_decay(self, ss=False):
        if ss:
            self.scheduler_ext.step()
            self.scheduler_dec.step()
        else:
            self.scheduler_ext.step()
            self.scheduler_reg.step()
            if self.multi:
                self.scheduler_cls.step()

    # def test(self, input, show=False):
    #     self.eval()
    #     self.forward(input, ss=False)
    #     self.train()
    #     self.error = torch.sqrt(
    #         (self.label[:, 0] - self.result[:, 0]) ** 2 + (self.label[:, 1] - self.result[:, 1]) ** 2)/2
    #     self.error += torch.sqrt(
    #         (self.label[:, 2] - self.result[:, 2]) ** 2 + (self.label[:, 3] - self.result[:, 3]) ** 2)/2
    #     self.error=torch.mean(self.error).detach().cpu()


    def heatmap2coord(self, heat_maps):
        gt_array = heat_maps[:, 0, :, :].view(-1, 160 * 160)
        idg = torch.argmax(gt_array, dim=1).cpu().float()
        xg1 = idg % 160
        yg1 = (idg - xg1) / 160
        gt_array = heat_maps[:, 1, :, :].view(-1, 160 * 160)
        idg = torch.argmax(gt_array, dim=1).cpu().float()
        xg2 = idg % 160
        yg2 = (idg - xg2) / 160
        return torch.stack((xg1,yg1,xg2,yg2),dim=1).cuda()


    def init_weights(self):
        nn.torch.nor

    def test(self, input, show=False):

        self.forward(input, ss=False)
        gt_array = self.label[:, 0, :, :].view(-1, 160 * 160)
        idg = torch.argmax(gt_array, dim=1).cpu().float()
        xg = idg % 160
        yg = (idg - xg) / 160

        if self.ff:
            xp=self.result[:,0].detach().cpu()
            yp=self.result[:,1].detach().cpu()
        else:
            pre_array = self.result[:, 0, :, :].view(-1, 160 * 160)
            idp = torch.argmax(pre_array, dim=1).cpu().float()
            xp = idp % 160
            yp = (idp - xp) / 160


        e = torch.sqrt((xp - xg) * (xp - xg) + (yp - yg) * (yp - yg))
        self.error = torch.mean(e)
        # if self.out_ch == 2:
        gt_array = self.label[:, 1, :, :].view(-1, 160 * 160)
        idg = torch.argmax(gt_array, dim=1).cpu().float()
        xg = idg % 160
        yg = (idg - xg) / 160

        if self.ff:
            xp=self.result[:,2].detach().cpu()
            yp=self.result[:,3].detach().cpu()
        else:
            pre_array = self.result[:, 1, :, :].view(-1, 160 * 160)
            idp = torch.argmax(pre_array, dim=1).cpu().float()
            xp = idp % 160
            yp = (idp - xp) / 160



        e = torch.sqrt((xp - xg) * (xp - xg) + (yp - yg) * (yp - yg))
        self.error = self.error * 0.5 + torch.mean(e) * 0.5
        if self.multi:
            self.c_error=self.accuracy_class()

    def save_net(self, save_path):
        torch.save(self.cpu().state_dict(), save_path)
        self.cuda()

    def load_net(self, save_path, ext_only=False):
        if ext_only:
            state_dict = torch.load(save_path)
            # s = {k: state_dict[k] for k in state_dict if 'extractor' in k}
            # self.extractor.load_state_dict(s, strict=False)
            self.extractor.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(torch.load(save_path))
