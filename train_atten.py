from Stud_Data_RI64 import myDataset
from models.PS_FCN_atten import PS_FCN as Atten
from Model64 import SUNET as Atten
import torch.utils.data as Data
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time

# trainloader=Data.DataLoader(md,batch_size=16,shuffle=True, num_workers=12)
torch.cuda.set_device(1)
total_epochs = 200
print_inter = 10
vali_inter = 200
validation_split = 0.2
shuffle_dataset = True
# stud_names = ['Nut_stud']
stud_names = ['Nut_stud', 'panel_stud', 'stud', 'ball_stud', 'T_stud']

for name in stud_names:
    net = Atten(in_ch=16).cuda()
    save_path = './checkpoints/' + name + '/RI64'
    md_train = myDataset('./mat/' + name + '/stud_data_RI_train.mat', aug=True, inch=16)
    md_test = myDataset('./mat/' + name + '/stud_data_RI_test.mat', aug=False, inch=16)


    mine = 100
    for epoch in range(total_epochs):
        train_loader = torch.utils.data.DataLoader(md_train, batch_size=8)
        for i, data in enumerate(train_loader):
            net(data)
            loss=net.update()
            if i%10==0:
                print('training epoch {} iter {} out of {}:   loss:{}, lr={}'.format(epoch,i,len(train_loader),loss,net.scheduler.get_lr()))
        error = []
        validation_loader = torch.utils.data.DataLoader(md_test, batch_size=2)
        for i, data in enumerate(validation_loader):
            net.test(data)
            error.append(net.error)
        e = torch.mean(torch.stack(error))
        if mine > e:
            mine = e
            net.save_net(save_path)
        if epoch % 1 == 0:
            print("epoch:{}, iter{}, Validation Error:{}".format(epoch, i, e))
        if epoch % 20 == 0:
            net.LR_decay()

    print(name, ' Finished. min_vali_error:', mine)
