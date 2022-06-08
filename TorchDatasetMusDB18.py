import os
import pickle
from numpy import ndarray
from random import randint
import torch.utils.data.dataset as dataset
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import pdb

class ProcessLRSRAudio:
    def __init__(self, nfft:int = 256, hop_size:int=128, upsample_ratio:int = 2) -> None:
        self.nfft:int = nfft
        self.hann_window:Tensor = torch.hann_window(nfft)
        self.hop_size:int = hop_size
        self.upsample_ratio:int = upsample_ratio
        self.upsample: nn.Module = torch.nn.Upsample(scale_factor=self.upsample_ratio, mode ='linear', align_corners = False)
    
    def low_sampled_audio(self,audio:Tensor) -> Tensor:
        original_time_size:int = audio.shape[-1]
        audio = F.pad(audio, (0, self.nfft), 'constant', 0)
        stft:Tensor = torch.stft(audio, self.nfft, self.hop_size, window=self.hann_window)
        stft[:,int((self.nfft//2+1) / self.upsample_ratio):,...] = 0.
        lowpass_audio:Tensor = torch.istft(stft, self.nfft, self.hop_size,window=self.hann_window)
        lowpass_audio = lowpass_audio[..., :original_time_size].detach()
        lowpass_audio = lowpass_audio[...,::self.upsample_ratio]
        return lowpass_audio
    
    def upsample_audio(self,audio:Tensor) -> Tensor:
        audio = audio.unsqueeze(1)
        audio = self.upsample(audio).squeeze(1)
        return audio
    
    def get_lr_hr_dict_keep_sr(self,hr_audio:Tensor) -> dict:
        low_audio = self.low_sampled_audio(hr_audio)
        low_audio = self.upsample_audio(low_audio)
        time_size:int = max(low_audio.shape[-1],hr_audio.shape[-1])
        return {"hr":hr_audio[...,:time_size],"lr":low_audio[...,:time_size]}
    
    def get_spectrogram_and_phase_from_audio(self,audio:torch, include_mask=False):
        stft:Tensor = torch.stft(audio, self.nfft, self.hop_size, window=self.hann_window, return_complex=True)
        spectrogram:Tensor = torch.abs(stft)
        phase:Tensor = torch.angle(stft)
        ret = {"spec":spectrogram, "phase": phase}
        if include_mask:
            mask = stft.clone()
            mask[:,int((self.nfft//2+1) / self.upsample_ratio):,...] = 0
            mask = torch.abs(mask)
            mask = torch.where(mask==0, 1, 0)
            ret['mask'] = mask
        return ret



class TorchDatasetMusDB18(dataset.Dataset):
    def __init__(self, dataset_dir:str, mode:str='HR', sr:int = 16000, 
                 segment_length_second:float=3, samples_per_track:int=64, need_mask=False) -> None:
        data_path_list:list = os.listdir(dataset_dir)
        self.data_set = list()

        for i,data_path in enumerate(data_path_list):
            print(f"{i+1}/{len(data_path_list)} {data_path}")
            self.data_set.append(self.read_feature_pickle(f"{dataset_dir}/{data_path}"))
        
        self.samples_per_track:int = samples_per_track
        self.segment_size = int(sr * segment_length_second)
        self.process = ProcessLRSRAudio()
        self.need_LR = mode=='LRHR'
        self.need_mask = need_mask
    
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
        
        #print(audio_dict["hr"].shape, audio_dict["lr"].shape)
        hr_dict = self.process.get_spectrogram_and_phase_from_audio(audio_dict['hr'], include_mask=True)
        hr_spec = hr_dict['spec'].float()
        mask = hr_dict['mask']
        #hr_spec = self.process.get_spectrogram_and_phase_from_audio(audio_dict["hr"])["spec"].float() # (1, 129, 376)
        lr_spec = self.process.get_spectrogram_and_phase_from_audio(audio_dict["lr"])["spec"].float()  
        bit = 32
        _, w, h = hr_spec.shape
        resize = torchvision.transforms.Resize((round(w/bit)*bit, round(h/bit)*bit), 
                                            torchvision.transforms.InterpolationMode.BILINEAR) 
        hr_spec = resize(hr_spec)
        lr_spec = resize(lr_spec)
        mask = resize(mask)
        result = {'HR' : hr_spec, 'SR' : lr_spec, 'Index' : index}
        if self.need_LR : 
            result['LR'] = lr_spec
        if self.need_mask:
            result['mask'] = mask
        
        return result
    
    
if __name__ == "__main__":
    data_set = TorchDatasetMusDB18("../data/musdb18_processed/train")    
    
    out = data_set[1]
    pdb.set_trace()