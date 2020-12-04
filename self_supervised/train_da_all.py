
import torch
import torch.utils.data as Data
from Model_hourglass_SSDAv2 import SUNET
from datasets.Stud_Data_alltype_aligned import myDataset

# stud  sample rate 50  spervised: nm only  6.0
# stud  sample rate 50  spervised: ri only  5.9
# stud  sample rate 50  spervised: ri +nm  6.0



# trainloader=Data.DataLoader(md,batch_size=16,shuffle=True, num_workers=12)
torch.cuda.set_device(0)
total_epochs = 501
print_inter = 10
vali_inter = 10
validation_split = 0.2
num_puzzle = 9
shuffle_dataset = True
# stud_names = ['Nut_stud']
stud_names = ['panel_stud', 'Nut_stud', 'T_stud', 'ball_stud', 'stud']
# num_puzzle=4:  54/22    93/88         67/68     76/81         86/86
# num_puzzle=9:  22/11    06/02         20/33     35/43         77/75

torch.cuda.empty_cache()
net = SUNET(in_ch=3, out_ch=2,ss=True, multitask=False, para_reduce=4).cuda()
path_train=['./mat/' + name + '/stud_data_train.mat' for name in stud_names]
path_test=['./mat/' + name + '/stud_data_test.mat' for name in stud_names]
path_train_ri=['./mat/' + name + '/stud_data_RI_train.mat' for name in stud_names]
path_test_ri=['./mat/' + name + '/stud_data_RI_test.mat' for name in stud_names]
md_train = myDataset(path_train,path_train_ri, aug=True, sample_rate=20, puzzle_num=num_puzzle, more_ri=True)
train_loader = torch.utils.data.DataLoader(md_train, batch_size=32, shuffle=True, num_workers=0)
load=False

# validation_loader = torch.utils.data.DataLoader(md_test, batch_size=8)
print('number of batches: {}'.format(len(train_loader)))

for epoch in range(total_epochs):
    train_loss_g = []
    train_loss_d = []
    train_loss_mn = []
    train_loss_ri = []
    train_loss_rip = []
    acc_gan = []
    acc_c = []
    acc_ri = []
    acc_rip = []
    acc_nm = []

    for i, data in enumerate(train_loader):
        net(data)
        if i%8<4:
            net.update_d()
            train_loss_d.append(net.Loss_d.detach().cpu())
            acc_gan.append(net.accuracy_gan())
        else:
            net.update_g(ss_only=False,multi=False, g_scale=min(epoch/300+0.1,0.8))
            train_loss_mn.append(net.Loss_rec_nm.detach().cpu())
            train_loss_ri.append(net.Loss_rec_ri.detach().cpu())
            train_loss_rip.append(net.Loss_rec_rip.detach().cpu())
            acc_ri.append(net.accuracy(domain='ri'))
            acc_rip.append(net.accuracy(domain='rip'))
            acc_nm.append(net.accuracy(domain='nm'))

            # domain adaptation results
            train_loss_g.append(net.Loss_g.detach().cpu())

            # dummy results
            # train_loss_g.append(net.Loss_rec_nm.detach().cpu())
            # train_loss_d.append(net.Loss_rec_nm.detach().cpu())
            # acc_gan.append(net.Loss_rec_nm.detach().cpu())
            # acc_c.append(net.Loss_rec_nm.detach().cpu())

        # if i%10<4:
        #     train_loss_g.append(net.Loss_g.detach().cpu())
        #     train_loss_d.append(net.Loss_d.detach().cpu())
        #     acc_gan.append(net.accuracy_gan())
        #     acc_c.append(net.accuracy_class())
        # else:
        #     train_loss_g.append(net.Loss_rec_nm.detach().cpu())
        #     train_loss_d.append(net.Loss_rec_nm.detach().cpu())
        #     acc_gan.append(net.accuracy())
        #     acc_c.append(net.accuracy())


        # train_loss_mn.append(net.Loss_rec_nm.detach().cpu())
        # train_loss_ri.append(net.Loss_rec_ri.detach().cpu())
        # acc_ri.append(net.accuracy(ri=True))
        # acc_nm.append(net.accuracy(ri=False))

    error = []


    if epoch%100==0:
        save_path = './checkpoints/' + 'all' + '/self_sup/net_stack_ssda_mul-dom{}.path'.format(epoch//100)
        # save_path = './checkpoints/' + 'all' + '/self_sup/net_stack_ss_mul-dom{}.path'.format(epoch//100)
        # save_path = './checkpoints/' + 'all' + '/self_sup/net_stack_ss_ri{}.path'.format(epoch//100)
        # save_path = './checkpoints/' + 'all' + '/self_sup/net_stack_ss_nm{}.path'.format(epoch//100)
        net.save_net(save_path)
    if (epoch+1) % 10 == 0:
        gl = torch.mean(torch.stack(train_loss_g))
        dl = torch.mean(torch.stack(train_loss_d))
        nml = torch.mean(torch.stack(train_loss_mn))
        ril = torch.mean(torch.stack(train_loss_ri))
        ripl = torch.mean(torch.stack(train_loss_rip))
        acr = torch.mean(torch.stack(acc_ri))
        acrp = torch.mean(torch.stack(acc_rip))
        acn = torch.mean(torch.stack(acc_nm))
        acg = torch.mean(torch.stack(acc_gan))
        # acc = torch.mean(torch.stack(acc_c))
        mine = 0
        print(
            "ep:{}, D_l:{:.3f}, G_l:{:.3f}, nm_l:{:.3f}, ri_l:{:.3f}, rip_l:{:.3f},acc_rip:{:.3f},acc_ri:{:.3f}, acc_nm:{:.3f}, acc_g:{:.3f}".format(
                epoch, dl, gl,
                nml, ril,ripl, acrp,acr, acn, acg))
        # print(
        #     "ep:{}, nm_l:{:.3f},  acc_nm:{:.3f}".format(
        #         epoch,          nml,acn))
    # if epoch % 20 == 0:
    #     net.LR_decay(ss=True)

print(' Finished. min_vali_error:', mine)
