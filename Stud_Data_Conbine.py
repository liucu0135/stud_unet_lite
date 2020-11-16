import torch.utils.data as Data
import h5py
import numpy as np
from Stud_Data_unlabel import myDataset_unlabel as nm_data
from Stud_Data_unlabel_ri import myDataset_unlabel as ri_data


class myDataset(Data.Dataset):
    def __init__(self, path_nm, path_ri, aug=False, inch=64, randp=False, puzzle_num=9, img_number=64):
        self.root_path_nm = path_nm
        self.root_path_ri = path_ri
        self.dataset_nm = nm_data(path_nm, aug=True,inch=3, puzzle_num=puzzle_num)
        self.dataset_ri = ri_data(path_ri,pnc=self.dataset_nm.pnc, aug=aug, puzzle_num=puzzle_num, img_number=img_number)

    def __len__(self):
        return len(self.dataset_ri)

    def __getitem__(self, ri_id):
        nm_id = ri_id % len(self.dataset_nm)
        return self.dataset_ri[ri_id], self.dataset_nm[nm_id]
