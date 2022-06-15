from ProcessLRSRAudio import ProcessLRSRAudio
import os
import torch
import pickle
from numpy import ndarray
from random import randint
import torch.utils.data.dataset as dataset
## EDITED by jaekwon im : wrote whole class
class TorchDatasetMusDB18SpecTest(dataset.Dataset):
    def __init__(self, dataset_dir:str, mode:str='HR', sr:int = 16000, scale_value:float = 65) -> None:
        data_path_list:list = os.listdir(dataset_dir)
        self.data_set = list()
        self.sr = sr

        for i,data_path in enumerate(data_path_list):
            print(f"{i+1}/{len(data_path_list)} {data_path}")
            self.data_set.append(self.read_feature_pickle(f"{dataset_dir}/{data_path}"))
        
        self.process = ProcessLRSRAudio()
        self.need_LR = True if mode == 'LRHR' else False
        self.scale_value = scale_value
    
    def __len__(self) -> int:
        return len(self.data_set)
    
    def read_feature_pickle(self, data_path) -> dict:
        with open(data_path, 'rb') as pickle_file:
            feature = pickle.load(pickle_file)
        return feature
    
    def __getitem__(self, index):
        audio:ndarray = self.data_set[index]["audio"][self.sr * 60 : self.sr * 63]
            
        data = torch.from_numpy(audio).unsqueeze(0)

        audio_dict:dict = self.process.get_lr_hr_dict_keep_sr(data) 

        hr_spec = self.process.get_spectrogram_and_phase_from_audio(audio_dict["hr"])["spec"].float()
        lr_spec = self.process.get_spectrogram_and_phase_from_audio(audio_dict["lr"])["spec"].float()

        hr_spec = hr_spec / self.scale_value
        lr_spec = lr_spec / self.scale_value

        result = {'HR' : hr_spec, 'SR' : lr_spec, 'Index' : index}
        if self.need_LR : 
            result['LR'] = lr_spec

        return result