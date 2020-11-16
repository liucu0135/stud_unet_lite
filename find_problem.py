from Stud_Data import myDataset
from Model import SUNET
import torch.utils.data as Data
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time

torch.cuda.set_device(1)
validation_split = 0.2
shuffle_dataset = True
stud_names = ['stud']
# stud_names = ['Nut_stud', 'panel_stud', 'stud', 'T_stud', 'ball_stud']

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

for name in stud_names:
    net = SUNET(in_ch=3, out_ch=2).cuda()
    load_path = './checkpoints/' + name + '/' + 'NM' + '/net_ss1.path'
    md_test = myDataset('./mat/' + name + '/stud_data_test.mat', aug=False, inch=3)
    validation_loader = torch.utils.data.DataLoader(md_test, shuffle=False, batch_size=1)
    mine = 100
    net.load_state_dict(torch.load(load_path))
    for i, data in enumerate(validation_loader):
        net.train()
        net(data)
        print(torch.mean(net.result.cpu().detach()[:,:,:,100:]))
        net(data)
        print(torch.mean(net.result.cpu().detach()[:,:,:,100:]))
