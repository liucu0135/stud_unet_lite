from Stud_Data_RI1 import myDataset
from Model_hourglass import SUNET
import torch.utils.data as Data
import torch
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time

# trainloader=Data.DataLoader(md,batch_size=16,shuffle=True, num_workers=12)
torch.cuda.set_device(1)
total_epochs = 640
print_inter = 10
vali_inter = 200
validation_split = 0.2
shuffle_dataset = True
stud_names = [ 'T_stud', 'T_stud', 'T_stud', 'T_stud', 'T_stud', 'T_stud']
# stud_names = ['Nut_stud', 'panel_stud', 'stud', 'T_stud', 'ball_stud']
mine = 100
for name in stud_names:

    torch.cuda.empty_cache()
    net = SUNET(in_ch=3, out_ch=2).cuda()
    save_path = './checkpoints/' + name + '/RI1_HG'
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

    train_loader = torch.utils.data.DataLoader(md_train, batch_size=12)
    validation_loader = torch.utils.data.DataLoader(md_test, batch_size=12)


    for epoch in range(total_epochs):
        train_loss = []
        for i, data in enumerate(train_loader):
            # if i % 10 == 0:
            #     print(i, 'of ', len(train_loader), 'done')
            net(data)
            net.update()
            train_loss.append(net.Loss.detach().cpu())
        error = []

        if epoch % 2 == 0:
            net.eval()
            torch.cuda.empty_cache()
            for i, data in enumerate(validation_loader):
                net.test(data)
                error.append(net.error)
            net.train()
            e = torch.mean(torch.stack(error))
            tl = torch.mean(torch.stack(train_loss))
            if mine > e:
                mine = e
                net.save_net(save_path)
        if epoch % 50 == 0:
            print("epoch:{}, train_loss{},Validation Loss:{}, Validation Error:{}".format(epoch, tl, net.Loss.data, e))
        if epoch % 96 == 0:
            net.LR_decay()

    print(name, ' Finished. min_vali_error:', mine)
