import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import json
import numpy as np
from video_summary.fragments import compute_fragments

class VideoData(Dataset):
    def __init__(self, name, mode, split_index, action_state_size):
        self.mode = mode
        self.name = name # 'tvsum' or 'summe'
        self.datasets = ['dataset/SumMe/eccv16_dataset_summe_google_pool5.h5',
                         'dataset/TVSum/eccv16_dataset_tvsum_google_pool5.h5']
        self.splits_filename = ['dataset/splits/' + self.name + '_splits.json']
        self.split_index = split_index # current split (varies from 0 to 4)

        if name == 'summe':
            self.filename = self.datasets[0]
        elif name == 'tvsum':
            self.filename = self.datasets[1]

        hdf = h5py.File(self.filename, 'r')
        self.action_fragments = {}
        self.list_features = []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split

        for video_name in self.split[self.mode + '_keys']:
            features = torch.Tensor(np.array(hdf[video_name + '/features']))
            self.list_features.append(features)
            self.action_fragments[video_name] = compute_fragments(features.shape[0], action_state_size)

        hdf.close()

    def __len__(self):
        self.len = len(self.split[self.mode + '_keys'])
        return self.len

    # In "train" mode it returns the features and the action_fragments
    # In "test" mode it also returns the video_name
    def __getitem__(self, index):
        video_name = self.split[self.mode + '_keys'][index]  # gets the current video name
        frame_features = self.list_features[index]

        if self.mode == 'test':
            return frame_features, video_name, self.action_fragments[video_name]
        else:
            return frame_features, self.action_fragments[video_name]

class CustomVideoData(Dataset):
    def __init__(self, mode, filepath, action_state_size):
        self.mode = mode
        self.filename = filepath

        hdf = h5py.File(self.filename, 'r')
        self.keys = list(hdf)

        self.action_fragments = {}
        self.list_features = []

        for key in self.keys:
            features = torch.Tensor(np.array(hdf[key + '/features']))
            self.list_features.append(features)
            self.action_fragments[key] = compute_fragments(features.shape[0], action_state_size)

        hdf.close()

    def __len__(self):
        self.len = len(self.keys)
        return self.len

    # In "train" mode it returns the features and the action_fragments
    # In "test" mode it also returns the video_name
    def __getitem__(self, index):
        video_name = self.keys[index]  # gets the current video name
        frame_features = self.list_features[index]

        if self.mode == 'test':
            return frame_features, video_name, self.action_fragments[video_name]
        else:
            return frame_features, self.action_fragments[video_name]

def get_loader(name, mode, split_index, action_state_size):
    vd = VideoData(name, mode, split_index, action_state_size)
    if mode.lower() == 'train':
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return vd

def get_loader_custom_video_data(mode, save_path, action_state_size):
    vd = CustomVideoData(mode, save_path, action_state_size)
    if mode.lower() == 'train':
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return vd