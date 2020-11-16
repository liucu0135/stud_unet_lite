## Nut 3.5@50,2.5@100, 2.0@200,1.6
## panel 3.8,3.3,2.8,4,1.9

from Stud_Data_RI64 import myDataset
from Model64 import SUNET
import torch.utils.data as Data
import torch
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
# stud_names = ['Nut_stud']
stud_names = ['Nut_stud', 'panel_stud', 'stud', 'ball_stud', 'T_stud']

for name in stud_names:
    net = SUNET(in_ch=32, out_ch=2).cuda()
    save_path = './checkpoints/' + name + '/RI64'
    md_train = myDataset('./mat/' + name + '/stud_data_RI_train.mat', aug=True, inch=32)
    md_test = myDataset('./mat/' + name + '/stud_data_RI_test.mat', aug=False, inch=32)



    mine = 100
    for epoch in range(total_epochs):
        train_loader = torch.utils.data.DataLoader(md_train, batch_size=4)
        for i, data in enumerate(train_loader):
            net(data)
            net.update()
        error = []
        train_loader=None
        torch.cuda.empty_cache()
        validation_loader = torch.utils.data.DataLoader(md_test, batch_size=2)
        for i, data in enumerate(validation_loader):
            net.test(data)
            error.append(net.error.detach().cpu())
        e = torch.mean(torch.stack(error))
        validation_loader=None
        torch.cuda.empty_cache()
        if mine > e:
            mine = e
            net.save_net(save_path)
        if epoch % 1 == 0:
            print("epoch:{}, iter{},Validation Loss:{}, Validation Error:{}".format(epoch, i, net.Loss.data, e))
        if epoch % 50 == 0:
            net.LR_decay()

    print(name, ' Finished. min_vali_error:', mine)
