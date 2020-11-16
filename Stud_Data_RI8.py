import torch.utils.data as Data
import h5py
import numpy as np


class myDataset(Data.Dataset):
    def __init__(self, path, aug, inch=8, randp=False):
        self.root_path = path
        self.inch = inch
        self.aug = aug
        self.ch_out = read_ch(path, 'output')
        self.read_data_to_memory()
        self.len = read_length(path, 'output')
        self.randp=randp
        if aug:
            self.len *= 4

    def read_data_to_memory(self):
        f = h5py.File(self.root_path, 'r')
        dataf = f.get('input')
        self.input = np.array(dataf, dtype=np.float32)
        dataf = f.get('output')
        self.output = np.array(dataf, dtype=np.float32)
        print('data "{}" loaded in RAM!'.format(self.root_path))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.aug:
            aug = item % 4
            item = (item - aug) // 4
        else:
            aug = 0

        if self.inch<64:
            if self.randp:
                rp=np.random.permutation(64)[:self.inch]
                input = self.input[item, rp, :, :]
            else:
                # input = self.input[item, ::2, ::2, ::2]
                input = self.input[item, ::4, :, :]
                # output = self.output[item, :, ::2, ::2]
                output = self.output[item, :, :, :]
        else:
            input = self.input[item, :, :, :]
            output = self.output[item, :, :, :]

        if aug % 2 == 1:
            input = np.ascontiguousarray(np.flip(input, 1))
            output = np.ascontiguousarray(np.flip(output, 1))
        if aug > 1:
            input = np.ascontiguousarray(np.flip(input, 2))
            output = np.ascontiguousarray(np.flip(output, 2))

        return input, output


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
