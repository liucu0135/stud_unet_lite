import torch.utils.data as Data
import h5py
import numpy as np
from datasets.Stud_Data_unlabel_alltype import myDataset_unlabel as nm_data
from datasets.Stud_Data_unlabel_ri_alltype import myDataset_unlabel as ri_data
import matplotlib.pyplot as plt









class myDataset(Data.Dataset):
    def __init__(self, path_nm, path_ri, aug=False, inch=64, randp=False, puzzle_num=9, img_number=1, sample_rate=1, more_ri=False):
        self.root_path_nm = path_nm
        self.root_path_ri = path_ri
        self.dataset_nm = nm_data(path_nm, aug=aug,inch=3, puzzle_num=puzzle_num, sample_rate=sample_rate, original_img=True )
        self.dataset_ri = ri_data(path_ri,pnc=self.dataset_nm.pnc, aug=aug, puzzle_num=puzzle_num, img_number=img_number, sample_rate=sample_rate, original_img=True)


    def __len__(self):
        return len(self.dataset_ri)

    def __getitem__(self, id):
        id_short=id%len(self.dataset_ri)
        return self.dataset_nm[id_short], self.dataset_ri[id_short]
