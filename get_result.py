from Stud_Data import myDataset as DS_mn
from Stud_Data_RI1 import myDataset as DS_ri1
from Stud_Data_RI import myDataset as DS_ri
from Stud_Data_RI64 import myDataset as DS_ri64
from Model import SUNET
import torch.utils.data as Data
import torch
import scipy.io as io
import numpy as np

# trainloader=Data.DataLoader(md,batch_size=16,shuffle=True, num_workers=12)
torch.cuda.set_device(0)
total_epochs = 10
print_inter = 10
vali_inter = 200
validation_split = 0.2
shuffle_dataset = True

stud_names = ['Nut_stud', 'panel_stud', 'T_stud', 'stud', 'ball_stud']
set_names = ['NM', 'RI', 'RI1', 'RI64']
in_ch_nums = {set_names[0]: 3, set_names[1]: 1, set_names[2]: 1, set_names[3]: 64}
data_sets = {set_names[0]: DS_mn, set_names[1]: DS_ri, set_names[2]: DS_ri1, set_names[3]: DS_ri64}
torch.cuda.empty_cache()

# stud_names = ['T_stud', 'stud', 'panel_stud', 'ball_stud', 'Nut_stud']
# data_sets = [DS_mn, DS_ri, DS_ri1, DS_ri64]

for name in stud_names[:]:
    for sname in set_names[:]:
        net = SUNET(in_ch=in_ch_nums[sname], out_ch=2).cuda()
        load_path = './checkpoints/' + name + '/' + sname + '/net_ss1.path'
        save_path = './checkpoints/' + name + '/' + sname + '_result.mat'
        net.load_state_dict(torch.load(load_path))
        # print("set:{}, data:{}, state_dict loaded. ".format(name, sname))
        if 'RI' in sname:
            md_test = data_sets[sname]('./mat/' + name + '/stud_data_RI_test.mat', aug=False, inch=in_ch_nums[sname])
        else:
            md_test = data_sets[sname]('./mat/' + name + '/stud_data_test.mat', aug=False, inch=in_ch_nums[sname])
        # print("set:{}, data:{}, data read. ".format(name, sname))
        validation_loader = torch.utils.data.DataLoader(md_test, batch_size=32)
        error = []
        predicted_result = []
        gt = []
        input_im = []
        for i, data in enumerate(validation_loader):
            # print('doing{} out of {} batches'.format(i, validation_loader.__len__()))
            net.test(data)
            # cpu_result = net.result.cpu().detach()
            error.append(net.error.cpu().detach())
            gt.append(net.label.cpu().detach())
            input_im.append(net.input.cpu().detach())
            predicted_result.append( net.result.cpu().detach())

        e = torch.mean(torch.stack(error))
        result = torch.cat(predicted_result, dim=0)
        gt = np.array(torch.cat(gt, dim=0))
        input_im = np.array(torch.cat(input_im, dim=0))
        result = np.array(result)
        e = np.array(e)



        io.savemat(save_path, {'result': result.transpose([2,3,1,0]), 'ave': e, 'gt': gt.transpose([2,3,1,0]), 'input': input_im.transpose([2,3,1,0])})
        print("set:{}, data:{}, has finished. Error:{}".format(name, sname, e))
