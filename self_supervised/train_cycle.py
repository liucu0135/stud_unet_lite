from Stud_Data import myDataset
from Stud_Data_unlabel import myDataset_unlabel
# from Model_hourglass_SS import SUNET
from Res18_ss_cycle import SUNET
import torch.utils.data as Data
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time

## nm train SS only
#stud  sample rate 10  spervised: 3.32
#stud  sample rate 50  spervised: 5.4
#stud  sample rate 100  spervised: 6.9


##  Benchmarks
#stud  sample rate 10  spervised: 6.1618  label_number: 46
#stud  sample rate 50  spervised: 4.7  label_number: 46
#stud  sample rate 100  spervised: 7.8636  label_number: 4

## nm SS only
#stud  sample rate 10  spervised: 3.69
#stud  sample rate 50  spervised: 5.88
#stud  sample rate 100  spervised: 8.3738



## SSDA
#stud  sample rate 10  spervised: 2.85
#stud  sample rate 50  spervised: 4.8
#stud  sample rate 100  spervised: 6.92


## ri SS only
#stud  sample rate 10  spervised: 2.85
#stud  sample rate 50  spervised: 5.43
#stud  sample rate 100  spervised: 7.66(6.97)



# trainloader=Data.DataLoader(md,batch_size=16,shuffle=True, num_workers=12)
torch.cuda.set_device(1)
total_epochs = 500
print_inter = 10
vali_inter = 200
validation_split = 0.2
shuffle_dataset = True
stud_names = ['stud']
# stud_names = ['panel_stud', 'Nut_stud',  'ball_stud', 'T_stud', 'stud']
for name in stud_names:
    mine = 100
    torch.cuda.empty_cache()
    net = SUNET(ss=False).cuda()
    save_path = './checkpoints/' + name + '/self_sup/net_ss_cycle0.path'
    # load_path = './checkpoints/' + name + '/self_sup/net_ss_only.path'
    load_path = './checkpoints/' + name + '/self_sup/net_ss_cycle.path'
    # load_path = './checkpoints/' + name + '/self_sup/net_ss_da0.path'
    sample_rate=200
    if name == 'stud':
        sample_rate=10
    md_train = myDataset('./mat/' + name + '/stud_data_train.mat', aug=False, inch=3, sample_rate=sample_rate)
    # md_train = myDataset_unlabel('./mat/' + name + '/stud_data_train.mat', aug=False, inch=3)
    md_test = myDataset('./mat/' + name + '/stud_data_test.mat', aug=False, inch=3)
    # md_test = myDataset_unlabel('./mat/' + name + '/stud_data_test.mat', aug=False, inch=3)
    load=True

    if load:
        load_path = './checkpoints/' + name + '/self_sup/net_ss_cycle.path'
        net.load_net(load_path, ext_only=True)
        # net.load_net(load_path)

    train_loader = torch.utils.data.DataLoader(md_train, batch_size=8, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(md_test, batch_size=8)


    for epoch in range(total_epochs):
        train_loss = []
        for i, data in enumerate(train_loader):
            # if i % 10 == 0:
            #     print(i, 'of ', len(train_loader), 'done')
            net(data, ss=False)
            net.update(ss=False, ext_fix=False)
            train_loss.append(net.Loss.detach().cpu())
        error = []
        V_loss = []

        if epoch % 10 == 0:
            net.eval()
            torch.cuda.empty_cache()
            for i, data in enumerate(validation_loader):
                net.test(data)
                error.append(net.error)
                V_loss.append(net.Loss.data)
            net.train()
            e = torch.mean(torch.stack(error))
            tl = torch.mean(torch.stack(train_loss))
            vl = torch.mean(torch.stack(V_loss))
            if mine > e:
                mine = e
                net.save_net(save_path)
        if epoch % 10 == 0:
            print("ep:{}, T_loss:{:2f},V_Loss:{:2f}, V_Error:{:2f}".format(epoch, tl, vl, e))
        if epoch % 50 == 0:
            net.LR_decay()

    print(name, ' Finished. min_vali_error:', mine)
