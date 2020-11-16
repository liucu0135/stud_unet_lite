import torch.utils.data as Data
import h5py
import numpy as np
from Stud_Data_alltype import myDataset as Dataset


class myDataset(Data.Dataset):
    def __init__(self, paths_nm, paths_ri, aug=False, inch=64, randp=False, puzzle_num=9, img_number=64, sample_rate=1):
        self.datasets_nm=[]
        self.datasets_ri=[]
        for pn, pr in zip(paths_nm, paths_ri):
            self.datasets_nm.append(Dataset(pn, aug=True,inch=3, puzzle_num=puzzle_num))
            self.datasets_ri.append(Dataset(paths_ri,pnc=self.dataset_nm.pnc, aug=aug, puzzle_num=puzzle_num, img_number=img_number))
        self.lens=[0]
        self.len=0
        for p in self.datasets_ri:
            if sample_rate==1:
                self.len += len(p)
            else:
                self.len += sample_rate
            self.lens.append(self.len)

    def __len__(self):
        return len(self.dataset_ri)

    def __getitem__(self, item0):
        if self.aug:
            aug = item0 % 4
            item0 = (item0 - aug) // 4
        else:
            aug = 0

        for c in range(len(self.lens)):
            if item0>=self.lens[c]:
                continue
            else:
                item=item0-self.lens[c-1]
                input_data=self.input[c-1]
                output_data=self.output[c-1]
                type=c-1
                break
        nm_id = item % len(self.dataset_nm)


        return self.datasets_ri[type][item], self.datasets_nm[type][nm_id]
