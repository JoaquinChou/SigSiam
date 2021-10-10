import torch
from torch.utils.data.dataset import Dataset
from utils.util import SignalTransform
import numpy as np
import os


class MotorSignalDataset(Dataset):
    def __init__(self, data_dir):
        super(MotorSignalDataset, self).__init__()
        signal_data = []
        labels_list = {
            'N': 0,
            'BF': 1,
            'BRBF': 2,
            'SWF': 3,
            'ESF': 4,
            'RUF': 5,
        }

        for class_dir in os.listdir(os.path.join(data_dir)):
            for filename in os.listdir(os.path.join(data_dir, class_dir)):
                signal_data.append(
                    os.path.join(data_dir, class_dir + '/', filename))

        self.signal_data = signal_data
        self.labels_list = labels_list

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, index):
        np_signal_data = np.load(self.signal_data[index])
        np_signal_data = np_signal_data.reshape(
            (1, np_signal_data.shape[0], 1))
        label_name = self.signal_data[index].split('/')[-2]
        transform = SignalTransform(np_signal_data)
        trans_signals = transform()
        label = self.labels_list[label_name]

        return trans_signals, np.array(label)


class NoTransformMotorSignalDataset(Dataset):
    def __init__(self, data_dir):
        super(NoTransformMotorSignalDataset, self).__init__()
        signal_data = []
        labels_list = {
            'N': 0,
            'BF': 1,
            'BRBF': 2,
            'SWF': 3,
            'ESF': 4,
            'RUF': 5,
        }

        for class_dir in os.listdir(os.path.join(data_dir)):
            for filename in os.listdir(os.path.join(data_dir, class_dir)):
                signal_data.append(
                    os.path.join(data_dir, class_dir + '/', filename))

        self.signal_data = signal_data
        self.labels_list = labels_list

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, index):
        np_signal_data = np.load(self.signal_data[index])
        np_signal_data = np_signal_data.reshape(
            (1, np_signal_data.shape[0], 1))
        label_name = self.signal_data[index].split('/')[-2]
        label = self.labels_list[label_name]

        np_signal_data = (np_signal_data - np.min(np_signal_data)) / (np.max(np_signal_data) - np.min(np_signal_data))
        
        return np_signal_data, np.array(label)
