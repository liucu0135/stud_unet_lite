import torch.utils.data as Data
import h5py
import numpy as np
from ultis import PN_converter
from numpy import math
import matplotlib.pyplot as plt

class myDataset_unlabel(Data.Dataset):
    def __init__(self, path, aug, pnc=None, puzzle_num=0, img_number=1):
        self.root_path = path
        self.ch_out = read_ch(path, 'output')
        self.len = read_length(path, 'output')
        self.read_data_to_memory()
        self.puzzle_num = int(np.sqrt(puzzle_num))
        self.img_num=img_number
        if pnc is None:
            self.pnc = PN_converter(puzzle_num,
                                pnumber=min(math.factorial(puzzle_num), 50))
        else:
            self.pnc=pnc
        if aug:
            self.aug = 4
        else:
            self.aug = 1

    def read_data_to_memory(self):
        f = h5py.File(self.root_path, 'r')
        dataf = f.get('input')
        self.input = np.array(dataf, dtype=np.float32)
        dataf = f.get('output')
        self.output = np.array(dataf, dtype=np.float32)
        print('data "{}" loaded in RAM!'.format(self.root_path))

    def __len__(self):
        return self.len * self.aug * self.img_num

    def __getitem__(self, item):
        # light = item % self.img_num
        light = item % self.img_num
        item = (item - light) // self.img_num
        light = 27
        if self.aug:
            aug = item % 4
            item = (item - aug) // 4
        else:
            aug = 0


        input = self.input
        input = np.expand_dims(input[item, light, :, :], axis=0)
        input= input/np.max(input)*2-1
        output = self.output[item, :, :, :]

        if aug % 2 == 0:
            input = np.ascontiguousarray(np.flip(input, 1))
            output = np.ascontiguousarray(np.flip(output, 1))
        if aug > 1:
            input = np.ascontiguousarray(np.flip(input, 2))
            output = np.ascontiguousarray(np.flip(output, 2))
        puzzle = np.repeat(input.copy(), 3, axis=0)
        puzzle_num = self.puzzle_num
        puzzle=np.zeros_like(puzzle)  # white back ground
        # idxs_array = np.random.permutation(puzzle_num * puzzle_num)
        self.perm_id = np.random.randint(min(math.factorial(puzzle_num**2), 50))
        idxs = self.pnc.num2perm(self.perm_id)
        #  hard coded for laziness

        if 'Nut' in self.root_path or 'panel' in self.root_path or 'T' in self.root_path:
            edge = 10
        else:
            edge = 10
        stride = (puzzle.shape[1] - edge * 2) // puzzle_num-edge//2
        for i in range(puzzle_num*puzzle_num):
            giggle=int(((np.random.rand()+1)*edge)//8)
            giggle2=int(((np.random.rand()+1)*edge)//8)
            col = i % puzzle_num
            row = (i - col) // puzzle_num
            tcol = idxs[i] % puzzle_num
            trow = (idxs[i] - tcol) // puzzle_num
            # puzzle[:, edge + trow * stride:edge + trow * stride + stride,
            # edge + tcol * stride:edge + tcol * stride + stride] = input[:,
            #                                                       edge + row * stride:edge + row * stride + stride,
            #                                                       edge + col * stride:edge + col * stride + stride]
            puzzle[:, edge//2 + trow * (stride+edge)+giggle:edge//2 + trow * (stride+edge) + stride+giggle,
            edge//2 + tcol * (stride+edge)+giggle2:edge//2 + tcol * (stride+edge) + stride+giggle2] = input[:,
                                                                  edge + row * stride+giggle2:edge + row * stride + stride+giggle2,
                                                                  edge + col * stride+giggle:edge + col * stride + stride+giggle]

        # plt.subplot(2,1,1)
        # plt.imshow(puzzle[0,:,:])
        # plt.subplot(2,1,2)
        # plt.imshow(input.squeeze())
        # plt.show()
        # print(self.perm_id)
        return puzzle, int(self.perm_id)


def read_input(filepath, key, s, light=-1):
    f = h5py.File(filepath, 'r')
    dataf = f.get(key)
    if light > -1:
        return np.expand_dims(np.array(dataf[s, light, :, :], dtype=np.float32), axis=0)
    return np.array(dataf[s, :, :, :], dtype=np.float32)


def read_output(filepath, key, item, ch_out):
    f = h5py.File(filepath, 'r')
    dataf = f.get(key)
    if ch_out == 1:
        return np.tile(np.array(dataf[item, 0, :, :], dtype=np.float32).squeeze(), (2, 1, 1))
    else:
        return np.array(dataf[item, :, :, :], dtype=np.float32).squeeze()


def read_ch(filepath, key):
    f = h5py.File(filepath, 'r')
    dataf = f.get(key)
    return np.array(dataf, dtype=np.float32).shape[1]


def read_length(filepath, key):
    f = h5py.File(filepath, 'r')
    dataf = f.get(key)
    return np.array(dataf, dtype=np.float32).shape[0]
