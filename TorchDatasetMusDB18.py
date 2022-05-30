import os
import pickle
from numpy import ndarray
from random import randint
import torch.utils.data.dataset as dataset

class TorchDatasetMusDB18(dataset.Dataset):
    def __init__(self, dataset_dir:str, sr:int = 16000, segment_length_second:float=3) -> None:
        data_path_list:list = os.listdir(dataset_dir)
        self.data_set = list()

        for i,data_path in enumerate(data_path_list):
            print(f"{i+1}/{len(data_path_list)} {data_path}")
            self.data_set.append(self.read_feature_pickle(f"{dataset_dir}/{data_path}"))
        
        self.samples_per_track:int = 64
        self.segment_size = int(sr * segment_length_second)
    
    def __len__(self) -> int:
        return self.samples_per_track * len(self.data_set)
    
    def read_feature_pickle(self, data_path) -> dict:
        with open(data_path, 'rb') as pickle_file:
            feature = pickle.load(pickle_file)
        return feature
    
    def __getitem__(self, index):
        index:int = index//self.samples_per_track
        
        total_time_samples:int = self.data_set[index]["audio"].shape[-1]
        segment_idx:int = randint(0, total_time_samples - self.segment_size)
        segmented_audio:ndarray = (self.data_set[index]["audio"][...,segment_idx:segment_idx+self.segment_size])
            
        return segmented_audio