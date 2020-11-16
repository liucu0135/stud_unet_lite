import torch
import torch.nn as nn
import torchvision.models as built_in_models
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class VGG16():
    def __init__(self, in_ch=3):
        self.net = built_in_models.vgg16(num_classes=4, pretrained=False)
        self.net.classifier[0] = nn.Linear(12800, 4096)
        self.net=self.net.cuda()
        self.criterion = nn.MSELoss()
        self.opt_all = optim.Adam(self.net.parameters(), lr=0.001)
        self.scheduler = StepLR(self.opt_all, step_size=1, gamma=0.5)

    def forward(self, data):
        # self.input = torch.cat((data[0],data[0],data[0]),dim=1).cuda()
        self.input=data[0].cuda()
        self.label = data[1]
        gt_array = self.label[:, 0, :, :].view(-1, 160 * 160)
        idg = torch.argmax(gt_array, dim=1).float()
        xg1 = idg % 160
        yg1 = (idg - xg1) / 160
        gt_array = self.label[:, 1, :, :].view(-1, 160 * 160)
        idg = torch.argmax(gt_array, dim=1).float()
        xg2 = idg % 160
        yg2 = (idg - xg2) / 160
        self.label=torch.stack([xg1,yg1,xg2,yg2],dim=1).cuda()
        self.result=self.net(self.input)

    def update(self):


        self.Loss = self.criterion(self.result.cuda(), self.label.detach())
        self.opt_all.zero_grad()
        self.Loss.backward()
        self.opt_all.step()

    def LR_decay(self):
        self.scheduler.step()

    def load_state_dict(self,sdict):
        self.net.load_state_dict(sdict)

    def cuda(self):
        self.net.cuda()

    def test(self, input):
        self.net.eval()
        self.forward(input)
        self.net.train()
        error = (self.result-self.label).detach().cpu()
        self.error =torch.mean(torch.sqrt(error[:,1]**2+error[:,0]**2)+torch.sqrt(error[:,3]**2+error[:,2]**2))/2
        # gt_array = self.label[:, 0, :, :].view(-1, 160 * 160)
        # pre_array = self.result[:, 0, :, :].view(-1, 160 * 160)
        # idp = torch.argmax(pre_array, dim=1).cpu().float()
        # idg = torch.argmax(gt_array, dim=1).cpu().float()
        # xp = idp % 160
        # yp = (idp - xp) / 160
        # xg = idg % 160
        # yg = (idg - xg) / 160
        # e = torch.sqrt((xp - xg) * (xp - xg) + (yp - yg) * (yp - yg))
        # self.error = torch.mean(e)
        # # if self.out_ch == 2:
        # gt_array = self.label[:, 1, :, :].view(-1, 160 * 160)
        # pre_array = self.result[:, 1, :, :].view(-1, 160 * 160)
        # idp = torch.argmax(pre_array, dim=1).cpu().float()
        # idg = torch.argmax(gt_array, dim=1).cpu().float()
        # xp = idp % 160
        # yp = (idp - xp) / 160
        # xg = idg % 160
        # yg = (idg - xg) / 160
        # e = torch.sqrt((xp - xg) * (xp - xg) + (yp - yg) * (yp - yg))
        # self.error = self.error * 0.5 + torch.mean(e) * 0.5

    def save_net(self, save_path):
        torch.save(self.net.cpu().state_dict(), save_path + '/net_ss1.path')
        self.net.cuda()