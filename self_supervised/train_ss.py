from Stud_Data import myDataset
from Stud_Data_unlabel import myDataset_unlabel
# from Stud_Data_unlabel_ri import myDataset_unlabel
from Model_hourglass_SS_cycle import SUNET
import torch.utils.data as Data
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time
#stud  sample rate 10  spervised: 3.32
#stud  sample rate 10  spervised: 2.7470
# trainloader=Data.DataLoader(md,batch_size=16,shuffle=True, num_workers=12)
torch.cuda.set_device(0)
total_epochs = 1000
print_inter = 10
vali_inter = 200
validation_split = 0.2
num_puzzle=9
shuffle_dataset = True
stud_names = ['stud']
# stud_names = ['Nut_stud', 'panel_stud', 'stud', 'T_stud', 'ball_stud']
mine = 100
for name in stud_names:
    torch.cuda.empty_cache()
    net = SUNET(in_ch=3, out_ch=2, num_puzzle=num_puzzle).cuda()
    save_path = './checkpoints/' + name + '/self_sup/net_ss_cycle.path'
    load_path = './checkpoints/' + name + '/self_sup/net_ss_cycle.path'
    md_train = myDataset('./mat/' + name + '/stud_data_train.mat', aug=False, inch=3)
    # md_train = myDataset_unlabel('./mat/' + name + '/stud_data_test.mat', aug=False, inch=3, puzzle_num=num_puzzle)
    # md_test = myDataset('./mat/' + name + '/stud_data_test.mat', aug=False, inch=3)
    md_test = myDataset('./mat/' + name + '/stud_data_test.mat', aug=False, inch=3)
    ss=True
    load=True
    if load:
        net.load_net(load_path, ext_only=False)

    train_loader = torch.utils.data.DataLoader(md_train, batch_size=8)


    for epoch in range(total_epochs):
        train_loss = []
        for i, data in enumerate(train_loader):
            net(data, ss)
            net.update(ss)
            train_loss.append(net.Loss.detach().cpu())
        error = []
        if epoch%10==0:

            net.show()
        if epoch % 1 == 0:
            tl = torch.mean(torch.stack(train_loss))
            if mine > tl:
                mine = tl
                net.save_net(save_path)
        if epoch % 1 == 0:
            print("ep:{}, T_loss:{:2f}".format(epoch, tl))
        if epoch % 100 == 0:
            net.LR_decay(ss)

