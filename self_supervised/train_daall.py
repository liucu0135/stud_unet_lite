from Stud_Data import myDataset
from Stud_Data_unlabel_ri import myDataset_unlabel
from Stud_Data_Conbine import myDataset
from Model_hourglass_SSDA import SUNET
import torch.utils.data as Data
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import time

# stud  sample rate 50  spervised: nm only  6.0
# stud  sample rate 50  spervised: ri only  5.9
# stud  sample rate 50  spervised: ri +nm  6.0



# trainloader=Data.DataLoader(md,batch_size=16,shuffle=True, num_workers=12)
torch.cuda.set_device(0)
total_epochs = 200
print_inter = 10
vali_inter = 200
validation_split = 0.2
num_puzzle = 9
shuffle_dataset = True
# stud_names = ['Nut_stud']
stud_names = ['panel_stud', 'Nut_stud', 'T_stud', 'ball_stud', 'stud']
# num_puzzle=4:  54/22    93/88         67/68     76/81         86/86
# num_puzzle=9:  22/11    06/02         20/33     35/43         77/75
for name in stud_names:
    mine = 100
    torch.cuda.empty_cache()
    net = SUNET(in_ch=3, out_ch=2, num_puzzle=num_puzzle, ss=True).cuda()
    save_path = './checkpoints/' + name + '/self_sup/net_ssda1.path'
    # save_path = './checkpoints/' + name + '/self_sup/net_ss_only_ri.path'
    # md_train = myDataset('./mat/' + name + '/stud_data_train.mat', aug=False, inch=3, sample_rate=10)
    md_train = myDataset('./mat/' + name + '/stud_data_test.mat', './mat/' + name + '/stud_data_RI_train.mat',
                         aug=True, inch=3, puzzle_num=num_puzzle, img_number=1)
    # md_train = myDataset('./mat/' + name + '/stud_data_test.mat','./mat/' + name + '/stud_data_RI_test.mat', aug=False, inch=3, puzzle_num=num_puzzle, img_number=8)
    # md_train = myDataset_unlabel('./mat/' + name + '/stud_data_RI_train.mat', aug=False, inch=3, puzzle_num=num_puzzle, img_number=8)
    # md_train = myDataset_unlabel('./mat/' + name + '/stud_data_RI_train.mat', aug=False, inch=3, puzzle_num=num_puzzle)
    # md_test = myDataset('./mat/' + name + '/stud_data_test.mat', aug=False, inch=3)
    # md_test = myDataset_unlabel('./mat/' + name + '/stud_data_test.mat', aug=False, inch=3)
    ss = True
    # load = False
    #
    # if load:
    #     net.load_net(load_path, ext_only=True)

    train_loader = torch.utils.data.DataLoader(md_train, batch_size=12)
    # validation_loader = torch.utils.data.DataLoader(md_test, batch_size=8)
    print('number of batches: {}'.format(len(train_loader)))

    for epoch in range(total_epochs):
        train_loss_g = []
        train_loss_d = []
        train_loss_mn = []
        train_loss_ri = []
        acc_gan = []
        acc_ri = []
        acc_nm = []

        for i, data in enumerate(train_loader):
            # if i % 1 == 0:
            #     net(data, g=False)
            # if i % 1 == 0:
            #     net.update_d()
            net(data, g=True)
            if epoch < 10000:
                net.update_g(ss_only=True)
            else:
                net.update_g(ss_only=False)

            if epoch < 10000:
                train_loss_g.append(net.Loss_rec_nm.detach().cpu())
                train_loss_d.append(net.Loss_rec_nm.detach().cpu())
                acc_gan.append(net.accuracy())
            else:
                train_loss_g.append(net.Loss_g.detach().cpu())
                train_loss_d.append(net.Loss_d.detach().cpu())
                acc_gan.append(net.accuracy_gan())

            # if not i % 2 == 1:
            #     train_loss_g.append(net.Loss_rec_nm.detach().cpu())
            # else:

            train_loss_mn.append(net.Loss_rec_nm.detach().cpu())
            # train_loss_ri.append(net.Loss_rec_nm.detach().cpu())
            train_loss_ri.append(net.Loss_rec_ri.detach().cpu())
            acc_ri.append(net.accuracy(ri=True))
            acc_nm.append(net.accuracy(ri=False))
            # acc_nm.append(net.accuracy(ri=False))

        error = []

        # net.show()
        if epoch % 1 == 0:
            # net.eval()
            # net.show()
            # torch.cuda.empty_cache()
            # for i, data in enumerate(validation_loader):
            #     net.test(data)
            #     error.append(net.error)
            # net.train()
            # e = torch.mean(torch.stack(error))
            gl = torch.mean(torch.stack(train_loss_g))
            dl = torch.mean(torch.stack(train_loss_d))
            nml = torch.mean(torch.stack(train_loss_mn))
            ril = torch.mean(torch.stack(train_loss_ri))
            acr = torch.mean(torch.stack(acc_ri))
            acn = torch.mean(torch.stack(acc_nm))
            acg = torch.mean(torch.stack(acc_gan))
            # if mine > nml + ril and epoch > 20:
            if epoch%50==0:
                save_path = './checkpoints/' + name + '/self_sup/net_ss_noad{}.path'.format(epoch//50)
                mine = nml + ril
                net.save_net(save_path)
        if epoch % 1 == 0:
            print(
                "ep:{}, D_l:{:1f}, G_l:{:1f}, nm_l:{:1f}, ri_l:{:1f}, acc_ri:{:1f}, acc_nm:{:1f}, acc_g:{:1f}".format(
                    epoch, dl, gl,
                    nml, ril, acr, acn, acg))
        if epoch % 20 == 0:
            net.LR_decay(ss)

    print(name, ' Finished. min_vali_error:', mine)
