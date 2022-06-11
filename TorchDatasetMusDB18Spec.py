from TorchDatasetMusDB18 import ProcessLRSRAudio
import os
import torch
import pickle
from glob import glob
from numpy import ndarray
from random import randint
import torch.utils.data.dataset as dataset

class TorchDatasetMusDB18Spec(dataset.Dataset):
    def __init__(self, dataset_dir:str, mode:str='HR', sr:int = 16000, segment_length_second:float=3, samples_per_track:int=64) -> None:
        data_path_list:list = glob(f"{dataset_dir}/*.pkl")
        self.data_set = list()

        for i,data_path in enumerate(data_path_list):
            print(f"{i+1}/{len(data_path_list)} {data_path}")
            self.data_set.append(self.read_feature_pickle(f"{data_path}"))
        
        self.samples_per_track:int = samples_per_track
        self.segment_size = int(sr * segment_length_second)
        self.process = ProcessLRSRAudio()
        self.need_LR = True if mode == 'LRHR' else False
    
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
            
        data = torch.from_numpy(segmented_audio).unsqueeze(0)

        audio_dict:dict = self.process.get_lr_hr_dict_keep_sr(data) # (1, 48000) x2

        hr_audio = audio_dict["hr"].unsqueeze(-2).float()
        lr_audio = audio_dict["lr"].unsqueeze(-2).float()

        # bit = 32
        # _, dur = hr_audio.sh
        # resize = torchvision.transforms.Resize((round(dur/bit)*bit), 
        #                                        torchvision.transforms.InterpolationMode.BILINEAR) 
        # hr_audio = resize(hr_audio)
        # lr_audio = resize(lr_audio)

        result = {'HR' : hr_audio, 'SR' : lr_audio, 'Index' : index}
        if self.need_LR : 
            result['LR'] = lr_audio

        return result