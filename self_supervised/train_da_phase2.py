from Stud_Data_alltype import myDataset
from Stud_Data_unlabel import myDataset_unlabel
from Model_hourglass_SSDAv2 import SUNET
import torch.utils.data as Data
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time

# 100% normal label: loss:0.0003   val_loss:0.00011  V_error: 1.3
# bench: 10% normal label: loss:25   val_loss:27  V_error: 16.2
# ss_ri: 10% normal label: loss:25   val_loss:27  V_error: 9.8
# ssdamu: 10% normal label: loss:29   val_loss:30  V_error: 8.3
# ssdamu+mu: 10% normal label: loss:29   val_loss:30  V_error: 8.0
# ss_ri: 10% normal label: loss:25   val_loss:27  V_error: 13.9
# ssda: 10% normal label: loss:29   val_loss:30  V_error: 8.7

# 10%
# ss_nm        8.0
# ss_ri        9.1
# ssda         7.2
# bench        10.19


# 5%
# ssda          10.3
# ssri          13.1
# ssnm          13.6
# bench          20






# 10% mu
# ss_ri           7.9
# ss_nm           8.2
# ssda           7.8




# ssdamu       10.0


# bench        11.2
# ss_ri        9.8
# ss_nm        9.6

# ssdamu       8.8
# ssdamu+mu    8.0
# ssnmmu+mu    6.9



# bench: 5% normal label: loss:0.0019   val_loss:0.0019  V_error: 9.5
# da-slr:5% normal label: loss:0.0042   val_loss:0.0037  V_error: 8.9




torch.cuda.set_device(0)
total_epochs = 600
print_inter = 10
vali_inter = 200
validation_split = 0.2
shuffle_dataset = True
# stud_names = ['Nut_stud']
stud_names = ['Nut_stud', 'panel_stud', 'stud', 'T_stud', 'ball_stud']
mine = 100
name='all'


torch.cuda.empty_cache()
net = SUNET(in_ch=3, out_ch=2, ss=False,para_reduce=4, scale_lr=1, ff=True, train_ext=True, multitask=False).cuda()
# save_path = './self_supervised/checkpoints/' + name + '/self_sup/net_temp.path'
save_path = './self_supervised/checkpoints/' + name + '/self_sup/net2_ss.path'
load_path = './self_supervised/checkpoints/' + name + '/self_sup/net_ss9.path'
path_train=['./mat/' + name + '/stud_data_train.mat' for name in stud_names]
path_test=['./mat/' + name + '/stud_data_test.mat' for name in stud_names]

md_train = myDataset(path_train, aug=True, inch=3, sample_rate=5)
md_test = myDataset(path_test, aug=False, inch=3)
load=True

if load:
    net.load_net(load_path, ext_only=True)
    # load_path = './self_supervised/checkpoints/' + name + '/self_sup/net2_ssda.path'
    # net.load_net(load_path)

train_loader = torch.utils.data.DataLoader(md_train, batch_size=10, shuffle=True)
validation_loader = torch.utils.data.DataLoader(md_test, batch_size=8)


if load:
    for epoch in range(5):
        train_loss = []
        c_loss = []
        for i, data in enumerate(train_loader):
            # if i % 10 == 0:
            #     print(i, 'of ', len(train_loader), 'done')
            net(data, ss=False)
            net.update(reg_only=True)
            train_loss.append(net.Loss.detach().cpu())
            c_loss.append(net.accuracy_class())
        print('pre train:  train_loss:{}, class_loss:{}'.format(torch.mean(torch.stack(train_loss)),torch.mean(torch.stack(c_loss))))


for epoch in range(total_epochs):
    train_loss = []
    c_loss = []
    for i, data in enumerate(train_loader):
        # if i % 10 == 0:
        #     print(i, 'of ', len(train_loader), 'done')
        net(data, ss=False)
        net.update(reg_only=False)
        train_loss.append(net.Loss.detach().cpu())
        c_loss.append(net.accuracy_class())
    error = []
    v_loss = []

    if epoch % 10 == 0:
        net.eval()
        torch.cuda.empty_cache()
        for i, data in enumerate(validation_loader):
            net.test(data)
            error.append(net.error)
            v_loss.append(net.accuracy_class())
        net.train()
        e = torch.mean(torch.stack(error))
        tl = torch.mean(torch.stack(train_loss))
        vl = torch.mean(torch.stack(v_loss))
        cl = torch.mean(torch.stack(c_loss))
        # if mine > e:
        #     mine = e
        #     net.save_net(save_path)
        net.save_net(save_path)
    if epoch % 10 == 0:
        print("ep:{}, T_loss:{:2f},class_acct:{:2f},class_accv:{:2f}, V_Error:{:2f}".format(epoch, tl,cl,vl, e))
    if epoch % 100 == 0:
        net.LR_decay()

print(name, ' Finished. min_vali_error:', mine)
