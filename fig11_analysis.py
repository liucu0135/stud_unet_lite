from Stud_Data import myDataset as DS_mn
from Stud_Data_RI1 import myDataset as DS_ri1
from Stud_Data_RI import myDataset as DS_ri
from Stud_Data_RI64 import myDataset as DS_ri64
from Model import SUNET
import torch.utils.data as Data
import torch
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np






# trainloader=Data.DataLoader(md,batch_size=16,shuffle=True, num_workers=12)
torch.cuda.set_device(1)
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


matset = []
for iname in range(5):
    matsubset = []
    for isname in range(3):
        name = stud_names[iname]
        sname = set_names[isname]
        save_path = './checkpoints/' + name + '/' + sname + '_result.mat'
        matsubset.append(io.loadmat(save_path))
    matset.append(matsubset)
plt.ion()
intensity_modifer=[3, 3, 6, 3, 15]
for i in range(10):
    fig, ax = plt.subplots(5, 3)
    for iname in range(5):
        mat_dict = matset[iname][0]
        its=intensity_modifer[iname]
        ave = mat_dict['ave'][0]
        gt = mat_dict['gt']
        result_nm = mat_dict['result']
        input_nm = mat_dict['input']

        mat_dict = matset[iname][1]
        result_ri = mat_dict['result']
        gt_ri = mat_dict['gt']

        mat_dict = matset[iname][2]
        result_ri1 = mat_dict['result']
        input_ri1 = mat_dict['input']

        # mat_dict = matset[iname][3]
        # result_ri64 = mat_dict['result']

        ind = np.unravel_index(np.argmax(gt[:, :, 0, i], axis=None), gt[:, :, 0, i].shape)
        ind_ri = np.unravel_index(np.argmax(gt_ri[:, :, 0, i], axis=None), gt_ri[:, :, 0, i].shape)
        ax[iname, 0].imshow(result_nm[:, :, 0, i])# predicted heat-map NM
        ax[iname, 0].scatter(ind[1],ind[0],c='r')

        ax[iname, 1].imshow(result_ri[:, :, 0, i])  # predicted heat-map RI
        ax[iname, 1].scatter(ind_ri[1], ind_ri[0],c='r')

        ax[iname, 2].imshow(result_ri1[:, :, 0, i])  # predicted heat-map RI1
        ax[iname, 2].scatter(ind[1], ind[0],c='r')

        # ax[iname, 6].imshow(result_ri64[:, :, 0, i])  # predicted heat-map RI1

    [axi.set_axis_off() for axi in ax.ravel()]
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.01)
    plt.show()
