import torch.utils.data as Data
import h5py
import numpy as np
from ultis import PN_converter
from numpy import math
import matplotlib.pyplot as plt


class myDataset_unlabel(Data.Dataset):
    def __init__(self, paths, aug, pnc=None, puzzle_num=0, img_number=1, sample_rate=1, original_img=False, more_ri=False):
        self.root_path = paths
        self.original_img = original_img
        self.inputs=[]
        self.outputs=[]
        self.lens=[0]
        self.len=0
        self.more_ri=more_ri
        for p in paths:
            self.read_data_to_memory(p)
            if sample_rate==1:
                self.len += read_length(p, 'output')
            else:
                self.len += sample_rate
            self.lens.append(self.len)
        self.ch_out = read_ch(paths[0], 'output')
        self.puzzle_num = int(np.sqrt(puzzle_num))
        self.img_num=img_number
        if pnc is None:
            self.pnc = PN_converter(puzzle_num,
                                pnumber=min(math.factorial(puzzle_num), 50))
        else:
            self.pnc=pnc
        if aug:
            self.aug = 1
            self.len*=4
        else:
            self.aug = 0

    def read_data_to_memory(self, path):
        f = h5py.File(path, 'r')
        dataf = f.get('input')
        # im2show = self.dataf[0,27,:,:]#.transpose([1, 2, 0])
        # plt.imshow(im2show)
        # plt.show()
        self.inputs.append(np.array(dataf, dtype=np.float32))
        dataf = f.get('output')
        self.outputs.append(np.array(dataf, dtype=np.float32))

        print('data "{}" loaded in RAM!'.format(path))

    def __len__(self):
        return self.len * self.img_num

    def __getitem__(self, item):
        # light = item % self.img_num
        # light = item % self.img_num
        # item = (item - light) // self.img_num
        light = 27
        # light = np.random.permutation(range(64))[:3]
        if self.aug:
            aug = item % 4
            item = (item - aug) // 4
        else:
            aug = 0

        for c in range(len(self.lens)):
            if item>=self.lens[c]:
                continue
            else:
                item=item-self.lens[c-1]
                self.input=self.inputs[c-1]
                self.output=self.outputs[c-1]
                type=c-1
                break

        input = self.input
        input = np.expand_dims(input[item, light, :, :], axis=0)


        # input = self.input[item,light,:,:]


        # input= input/np.max(input)
        input= input-np.mean(input)  # zero centered
        input= input/(np.std(input)+0.00000001)   # normalize variance

        output = self.output[item, :, :, :]
        # flip to match the normal map
        input=np.ascontiguousarray(np.flip(input, 1))
        output = np.ascontiguousarray(np.flip(output, 1))


        if aug % 2 == 0:
            input = np.ascontiguousarray(np.flip(input, 1))
            output = np.ascontiguousarray(np.flip(output, 1))
        if aug > 1:
            input = np.ascontiguousarray(np.flip(input, 2))
            output = np.ascontiguousarray(np.flip(output, 2))
        puzzle = np.repeat(input.copy(), 3, axis=0)
        puzzle_num = self.puzzle_num
        #puzzle=np.zeros_like(puzzle)  # white back ground
        # idxs_array = np.random.permutation(puzzle_num * puzzle_num)
        self.perm_id = np.random.randint(min(math.factorial(puzzle_num**2), 50))
        idxs = self.pnc.num2perm(self.perm_id)
        #  hard coded for laziness

        if 'Nut' in self.root_path[type] or 'panel' in self.root_path[type] or 'T' in self.root_path[type]:
            edge = 10
        else:
            edge = 10
        stride = (puzzle.shape[1] - edge * 2) // puzzle_num-edge//2
        puzzle_list = []
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
            puzzle_img=input[:,edge + row * stride+giggle2:edge + row * stride + stride+giggle2, edge + col * stride+giggle:edge + col * stride + stride+giggle]
            puzzle_list.append(np.repeat(puzzle_img, 3, axis=0))
            puzzle[:, edge//2 + trow * (stride+edge)+giggle:edge//2 + trow * (stride+edge) + stride+giggle,
            edge//2 + tcol * (stride+edge)+giggle2:edge//2 + tcol * (stride+edge) + stride+giggle2] =puzzle_img
        # plt.subplot(2,1,1)
        # plt.imshow(puzzle[0,:,:])
        # plt.subplot(2,1,2)
        # plt.imshow(input.squeeze())
        # plt.show()
        # print(self.perm_id)
        if self.more_ri:
            idx=np.random.randint(self.len)
            input = np.expand_dims(self.input[idx, np.random.randint(self.img_num), :, :], axis=0)
            input = input / np.max(input)

            return puzzle, int(self.perm_id), np.repeat(input, 3, axis=0), output
        if self.original_img:
            return puzzle, int(self.perm_id), np.repeat(input,3,axis=0), output
        else:
            return puzzle, int(self.perm_id), puzzle_list


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
