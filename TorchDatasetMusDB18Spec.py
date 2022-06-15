from ProcessLRSRAudio import ProcessLRSRAudio
import os
import torch
import pickle
from numpy import ndarray
from random import randint
import torch.utils.data.dataset as dataset

class TorchDatasetMusDB18Spec(dataset.Dataset):
    def __init__(self, dataset_dir:str, mode:str='HR', sr:int = 16000, segment_length_second:float=3, samples_per_track:int=64, scale_value:float = 65, need_mask=False, need_audio=False) -> None:
        data_path_list:list = os.listdir(dataset_dir)
        self.data_set = list()

        for i,data_path in enumerate(data_path_list):
            print(f"{i+1}/{len(data_path_list)} {data_path}")
            self.data_set.append(self.read_feature_pickle(f"{dataset_dir}/{data_path}"))
        
        self.samples_per_track:int = samples_per_track
        self.segment_size = int(sr * segment_length_second)
        self.process = ProcessLRSRAudio()
        self.need_LR = True if mode == 'LRHR' else False
        self.scale_value = scale_value
        self.need_mask = need_mask
        self.need_audio = need_audio
    
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
        #segmented_audio:ndarray = (self.data_set[index]["audio"][...,segment_idx:segment_idx+self.segment_size])
        segmented_audio:ndarray = (self.data_set[index]["audio"][..., 16000*60:16000*63])
            
        data = torch.from_numpy(segmented_audio).unsqueeze(0)

        audio_dict:dict = self.process.get_lr_hr_dict_keep_sr(data) # (1, 48000) x2
        
        hr_dict = self.process.get_spectrogram_and_phase_from_audio(audio_dict['hr'], include_mask=True)
        hr_spec = hr_dict['spec'].float()
        mask = hr_dict['mask']
        #hr_spec = self.process.get_spectrogram_and_phase_from_audio(audio_dict["hr"])["spec"].float() # (1, 129, 376)
        lr_spec = self.process.get_spectrogram_and_phase_from_audio(audio_dict["lr"])["spec"].float()

        hr_spec = hr_spec / self.scale_value
        lr_spec = lr_spec / self.scale_value

        result = {'HR' : hr_spec, 'SR' : lr_spec, 'Index' : index}
        if self.need_LR : 
            result['LR'] = lr_spec
        if self.need_mask:
            result['mask'] = mask
        if self.need_audio:
            result['audio'] = audio_dict['hr']

        return result