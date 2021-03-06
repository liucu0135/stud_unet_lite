from Model_hourglass_SSDA import SUNET
from Stud_Data_alltype import myDataset
from Stud_Data_unlabel import myDataset_unlabel
# from Model_hourglass_SS import SUNET
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
#stud  sample rate 50  spervised: 6.8636  label_number: 9
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
torch.cuda.set_device(0)
total_epochs = 200
print_inter = 10
vali_inter = 200
validation_split = 0.2
shuffle_dataset = True
# stud_names = ['Nut_stud']
stud_names = ['panel_stud', 'Nut_stud',  'ball_stud', 'T_stud', 'stud']
#  11.05  11.93  2.1  4.7  6.9

#  10.17  10.22  1.92  4.59  7.65
#  12.85  12.18  3.06  10.79  8.23
#  17.34  20.36  3.5  7.7  9.8
#  14.34  26.36  17 10.75  17.7  no bn, batch=16, scratch
#  11.34  11.36  1.8 4.2  7.4  bn, batch=16, scratch
#  11.34  11.36  2.3 4.5  6.4  bn, batch=16
tl = []
# for save_id in range(10):
for name in stud_names:
    save_id=3
    # name='Nut_stud'
    mine = 100
    torch.cuda.empty_cache()

    save_path = './checkpoints/' + name + '/self_sup/net_ss0.path'
    # load_path = './checkpoints/' + name + '/self_sup/net_ss_only.path'
    # load_path = './checkpoints/' + name + '/self_sup/net_ssda1.path'
    # load_path = './checkpoints/' + name + '/self_sup/net_ss_da0.path'
    sample_rate=10
    if name == 'stud':
        sample_rate=40
    md_train = myDataset('./mat/' + name + '/stud_data_train.mat', aug=True, inch=3, sample_rate=sample_rate)
    # md_train = myDataset_unlabel('./mat/' + name + '/stud_data_train.mat', aug=False, inch=3)
    md_test = myDataset('./mat/' + name + '/stud_data_test.mat', aug=False, inch=3)
    # md_test = myDataset_unlabel('./mat/' + name + '/stud_data_test.mat', aug=False, inch=3)
    load=True
    net = SUNET(in_ch=3, out_ch=2, ss=False, small_lr=load).cuda()
    if load:
        load_path = './checkpoints/' + name + '/self_sup/net_ss_nm{}.path'.format(save_id)
        net.load_net(load_path, ext_only=True)
        # net.load_net(save_path, ext_only=False)


        # net.load_net(load_path)

    train_loader = torch.utils.data.DataLoader(md_train, batch_size=12, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(md_test, batch_size=8)

    train_loss = []
    for epoch in range(total_epochs):

        for i, data in enumerate(train_loader):
            # if i % 10 == 0:
            #     print(i, 'of ', len(train_loader), 'done')
            net(data, ss=False)
            net.update()
            train_loss.append(net.Loss.detach().cpu())
        error = []
        vl = []
        if epoch % 10 == 0:
            net.eval()
            torch.cuda.empty_cache()
            for i, data in enumerate(validation_loader):
                net.test(data)
                error.append(net.error)
                net.cal_loss()
                vl.append(net.Loss.detach().cpu().data)
            net.train()
            e = torch.mean(torch.stack(error))
            tl = torch.mean(torch.stack(train_loss))
            vl = torch.mean(torch.stack(vl))
            if mine > e:
                mine = e
                # net.save_net(save_path)
            train_loss = []
        if epoch % 10 == 0:
            print("ep:{}, T_loss:{:2f},V_Loss:{:2f}, V_Error:{:2f}".format(epoch, tl, vl, e))
        if epoch % 50 == 0:
            net.LR_decay()

    print(name, ' Finished. min_vali_error:', mine)
