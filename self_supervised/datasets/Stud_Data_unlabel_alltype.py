import torch.utils.data as Data
import h5py
import numpy as np
import matplotlib.pyplot as plt
from ultis import PN_converter
from numpy import math


class myDataset_unlabel(Data.Dataset):
    def __init__(self, paths, aug, inch, sample_rate=1, puzzle_num=4, original_img=False):
        self.sample_rate = sample_rate
        self.inch = inch
        self.original_img = original_img
        self.aug = aug
        self.root_path = paths
        self.inputs = []
        self.outputs = []
        self.lens = [0]
        self.len = 0
        for p in paths:
            self.read_data_to_memory(p)
            if sample_rate == 1:
                self.len += read_length(p, 'output')
            else:
                self.len += sample_rate
            self.lens.append(self.len)

        self.ch_out = read_ch(paths[0], 'output')
        self.puzzle_num = int(np.sqrt(puzzle_num))
        self.pnc = PN_converter(puzzle_num, pnumber=min(math.factorial(puzzle_num), 50))
        if aug:
            self.len *= 4
        if inch == 1:
            self.len *= 64

    def read_data_to_memory(self, path):
        f = h5py.File(path, 'r')
        dataf = f.get('input')
        self.inputs.append(np.array(dataf, dtype=np.float32))
        dataf = f.get('output')
        self.outputs.append(np.array(dataf, dtype=np.float32))
        # if self.output.shape[1] == 1:
        #    self.output = np.array(dataf, dtype=np.float32)
        print('data "{}" loaded in RAM!'.format(path))

    def __len__(self):
        return self.len

    def __getitem__(self, item0):
        if self.aug:
            aug = item0 % 4
            item0 = (item0 - aug) // 4
        else:
            aug = 0

        for c in range(len(self.lens)):
            if item0 >= self.lens[c]:
                continue
            else:
                item = item0 - self.lens[c - 1]
                self.input = self.inputs[c - 1]
                self.output = self.outputs[c - 1]
                type = c - 1
                break

        if self.inch == 1:
            light = item % 64
            item = (item - light) // 64

        if self.inch == 1:
            input = np.expand_dims(self.input[item, light, :, :], axis=0)
        else:
            input = self.input[item, :, :, :]
        output = self.output[item, :, :, :]

        if aug % 2 == 1:
            # plt.imshow(input[0,:,:])
            # plt.show()
            input = np.ascontiguousarray(np.flip(input, 1))
            if self.inch == 3:
                input[0, :, :] *= -1
            output = np.ascontiguousarray(np.flip(output, 1))
        if aug > 1:
            input = np.ascontiguousarray(np.flip(input, 2))
            if self.inch == 3:
                input[1, :, :] *= -1
            output = np.ascontiguousarray(np.flip(output, 2))





        # recenter the image:
        idp=np.reshape(output[1, :, :],(-1))
        idp=np.argmax(idp)
        xp = idp % 160
        yp = (idp - xp) // 160
        idp=np.reshape(output[0, :, :],(-1))
        idp=np.argmax(idp)
        xp2 = idp % 160
        yp2 = (idp - xp2) // 160
        center = [(xp+xp2)//2,(yp+yp2)//2]
        # center =[xp,yp]
        center=np.array([c for c in center]).clip(40,120)
        input=input[:,center[1]-40:center[1]+40,center[0]-40:center[0]+40]

        puzzle = input.copy()
        puzzle_num = self.puzzle_num
        # puzzle = np.zeros_like(puzzle)  # white back ground
        # idxs_array = np.random.permutation(puzzle_num * puzzle_num)
        self.perm_id = np.random.randint(min(math.factorial(puzzle_num ** 2), 50))
        idxs = self.pnc.num2perm(self.perm_id)
        #  hard coded for laziness
        if 'Nut' in self.root_path[type] or 'panel' in self.root_path[type] or 'T' in self.root_path[type]:
            edge = 2
        else:
            edge = 2
        stride = (puzzle.shape[1] - edge * 2) // puzzle_num - edge // 2
        puzzle_list = []
        for i in range(puzzle_num * puzzle_num):
            giggle = 0#int(((np.random.rand() + 1) * edge) // 8)
            giggle2 = 0#int(((np.random.rand() + 1) * edge) // 8)
            col = i % puzzle_num
            row = (i - col) // puzzle_num
            tcol = idxs[i] % puzzle_num
            trow = (idxs[i] - tcol) // puzzle_num
            puzzle_img = input[:, edge + row * stride + giggle2:edge + row * stride + stride + giggle2,
                         edge + col * stride + giggle:edge + col * stride + stride + giggle]
            puzzle_list.append(puzzle_img)
            puzzle[:, edge // 2 + trow * (stride + edge) + giggle:edge // 2 + trow * (stride + edge) + stride + giggle,
            edge // 2 + tcol * (stride + edge) + giggle2:edge // 2 + tcol * (
                        stride + edge) + stride + giggle2] = puzzle_img
        # plt.imshow(np.transpose(input[:,center[1]-40:center[1]+40,center[0]-40:center[0]+40], (1,2,0)))
        # plt.imshow(np.transpose(puzzle, (1,2,0)))
        # plt.show()
        if self.original_img:
            return puzzle, int(self.perm_id), int(type), input, output, puzzle_list
        else:
            return puzzle, int(self.perm_id), puzzle_list


def read_input(filepath, key, s):
    f = h5py.File(filepath, 'r')
    dataf = f.get(key)
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
