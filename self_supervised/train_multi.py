from Stud_Data_alltype import myDataset
from Stud_Data_unlabel import myDataset_unlabel
from Model_hourglass_SSDA import SUNET
import torch.utils.data as Data
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time
#stud  sample rate 10  spervised: 6.1618
#stud  sample rate 50  spervised: 6.8636
#stud  sample rate 100  spervised: 7.8636
# trainloader=Data.DataLoader(md,batch_size=16,shuffle=True, num_workers=12)
torch.cuda.set_device(0)
total_epochs = 500
print_inter = 10
vali_inter = 200
validation_split = 0.2
shuffle_dataset = True
# stud_names = ['stud']
stud_names = ['Nut_stud', 'panel_stud', 'stud', 'T_stud', 'ball_stud']
mine = 100
name='all'


torch.cuda.empty_cache()
net = SUNET(in_ch=3, out_ch=2,ss=False, multitask=True, para_reduce=4).cuda()
save_path = './checkpoints/' + name + '/self_sup/net_da_multi.path'
load_path = './checkpoints/' + name + '/self_sup/net_da_multi.path'
path_train=['./mat/' + name + '/stud_data_train.mat' for name in stud_names]
path_test=['./mat/' + name + '/stud_data_test.mat' for name in stud_names]

md_train = myDataset(path_train, aug=True, inch=3, sample_rate=10)
md_test = myDataset(path_test, aug=False, inch=3)
load=False

if load:
    net.load_net(load_path)

train_loader = torch.utils.data.DataLoader(md_train, batch_size=8, shuffle=True)
validation_loader = torch.utils.data.DataLoader(md_test, batch_size=8)


for epoch in range(total_epochs):
    train_loss = []
    for i, data in enumerate(train_loader):
        # if i % 10 == 0:
        #     print(i, 'of ', len(train_loader), 'done')
        net(data, ss=False)
        net.update(multi=True)
        train_loss.append(net.Loss.detach().cpu())
    error = []
    c_error = []

    if epoch % 10 == 0:
        net.eval()
        torch.cuda.empty_cache()
        for i, data in enumerate(validation_loader):
            net.test(data)
            error.append(net.error)
            c_error.append(net.c_error)
        net.train()
        e = torch.mean(torch.stack(error))
        ec = torch.mean(torch.stack(c_error))
        tl = torch.mean(torch.stack(train_loss))
        if mine > e:
            mine = e
            net.save_net(save_path)
    if epoch % 10 == 0:
        print("ep:{}, T_loss:{:2f},V_Loss:{:2f}, V_Error:{:2f}, c_error:{:2f}".format(epoch, tl, net.Loss.data, e, ec))
    if epoch % 100 == 0:
        net.LR_decay()

print(name, ' Finished. min_vali_error:', mine)
