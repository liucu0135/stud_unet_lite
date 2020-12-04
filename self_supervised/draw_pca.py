
import torch
import torch.utils.data as Data
from Model_hourglass_SSDAv2 import SUNET
from datasets.Stud_Data_alltype_aligned_PCA import myDataset
from PCA_ploter import PCA_ploter
import numpy as np

# stud  sample rate 50  spervised: nm only  6.0
# stud  sample rate 50  spervised: ri only  5.9
# stud  sample rate 50  spervised: ri +nm  6.0



# trainloader=Data.DataLoader(md,batch_size=16,shuffle=True, num_workers=12)
torch.cuda.set_device(0)
total_epochs = 301
print_inter = 10
vali_inter = 10
validation_split = 0.2
num_puzzle = 9
shuffle_dataset = True
stud_names = ['Nut_stud']
# stud_names = ['panel_stud', 'Nut_stud', 'T_stud', 'ball_stud', 'stud']
# num_puzzle=4:  54/22    93/88         67/68     76/81         86/86
# num_puzzle=9:  22/11    06/02         20/33     35/43         77/75
for name in stud_names:
    checkpoints={'ri':'./checkpoints/' + 'all' + '/self_sup/net_stack_ss_ri{}.path'.format(2),
                 'ssda':'./checkpoints/' + 'all' + '/self_sup/net_stack_ssda_mul-dom{}.path'.format(2)
                 }
    for type in checkpoints:
        torch.cuda.empty_cache()
        net = SUNET(in_ch=3, out_ch=2,ss=True, multitask=False, para_reduce=4).cuda()
        path_train=['./mat/' + name + '/stud_data_train.mat' for name in stud_names]
        path_test=['./mat/' + name + '/stud_data_test.mat' for name in stud_names]
        path_train_ri=['./mat/' + name + '/stud_data_RI_train.mat' for name in stud_names]
        path_test_ri=['./mat/' + name + '/stud_data_RI_test.mat' for name in stud_names]
        md_train = myDataset(path_train,path_train_ri, aug=True, sample_rate=20, puzzle_num=num_puzzle, more_ri=True)
        train_loader = torch.utils.data.DataLoader(md_train, batch_size=32, shuffle=True, num_workers=0)

        net.load_net(checkpoints[type], ext_only=True)
        print('number of batches: {}'.format(len(train_loader)))

        fns=[]
        frs=[]
        for i, data in enumerate(train_loader):
            fn,fr=net(data, feat_only=True)
            fns.append(fn)
            frs.append(fr)
            print(torch.sum(fn-fr))
        svname='./self_supervised/plot/'+type+'_'+name+'_.png'
        fn=torch.cat(fns, dim=0)
        ln=torch.zeros(fn.shape[0])
        fr=torch.cat(frs, dim=0)
        lr=torch.ones(fr.shape[0])
        x=torch.cat((fn,fr))
        y=torch.cat((ln,lr))
        PCA_ploter(x.numpy().astype(np.float64),y.numpy().astype(np.int), save_name=svname)

        pass