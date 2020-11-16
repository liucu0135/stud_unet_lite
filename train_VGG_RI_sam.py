from Stud_Data_RI_sam import myDataset
from Model import SUNET
import torch.utils.data as Data
import torch
import os
from VGG import VGG16 as VGG
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time

# trainloader=Data.DataLoader(md,batch_size=16,shuffle=True, num_workers=12)
torch.cuda.set_device(0)
total_epochs = 640
print_inter = 10
vali_inter = 200
validation_split = 0.2
shuffle_dataset = True
stud_names = [ 'Nut_stud','panel_stud', 'stud', 'T_stud', 'ball_stud']
# stud_names = ['Nut_stud', 'panel_stud', 'stud', 'T_stud', 'ball_stud']

for name in stud_names:
    net = VGG()
    save_path = './checkpoints/' + name + '/RI_sam_VGG'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    md_train = myDataset('./mat/' + name + '/stud_data_RI_train.mat', aug=True, inch=3)
    md_test = myDataset('./mat/' + name + '/stud_data_RI_test.mat', aug=False, inch=3)
    # dataset_size = len(md)
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # if shuffle_dataset :
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]
    #
    # # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(md_train, batch_size=16)
    validation_loader = torch.utils.data.DataLoader(md_test, batch_size=16)

    mine = 100
    for epoch in range(total_epochs):
        for i, data in enumerate(train_loader):
            # if i % 10 == 0:
            #     print(i, 'of ', len(train_loader), 'done')
            net.forward(data)
            net.update()
        error = []
        for i, data in enumerate(validation_loader):
            net.test(data)
            error.append(net.error)
        e = torch.mean(torch.stack(error))
        if mine > e:
            mine = e
            net.save_net(save_path)
        if epoch % 10 == 0:
            print("epoch:{}, iter{},Validation Loss:{}, Validation Error:{}".format(epoch, i, net.Loss.data, e))
        if epoch % 96 == 0:
            net.LR_decay()

    print(name, ' Finished. min_vali_error:', mine)
