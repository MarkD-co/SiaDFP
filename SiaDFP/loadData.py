import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from config import config


class DiskDataset(Dataset):
    def __init__(self, train = True):
        self.data_path = '/home/kdma/disk/deal_data/' + config.disk_model + '/'
        if train:
            self.data = np.load(self.data_path+'/train/X_data.npy')
            self.label = np.load(self.data_path+'/train/y_labels.npy')
        else:
            self.data = np.load(self.data_path+'/test/X_data.npy')
            self.label = np.load(self.data_path + '/test/y_labels.npy')
    def __getitem__(self, index):
        data = self.data[index]
        half_row = int(data.shape[0]/2)
        part1 = data[0:half_row]
        part2 = data[half_row:]
        return torch.from_numpy(np.array(part1)), torch.from_numpy(np.array(part2)), torch.from_numpy(np.array(self.label[index]))

    def __len__(self):
        
        return self.data.shape[0]
