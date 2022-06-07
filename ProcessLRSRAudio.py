import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


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
        low_audio:Tensor = self.low_sampled_audio(hr_audio)
        low_audio = self.upsample_audio(low_audio)
        time_size:int = max(low_audio.shape[-1],hr_audio.shape[-1])
        return {"hr":hr_audio[...,:time_size],"lr":low_audio[...,:time_size]}
    
    def get_spectrogram_and_phase_from_audio(self,audio:torch):
        stft:Tensor = torch.stft(audio, self.nfft, self.hop_size, window=self.hann_window, return_complex=True)
        spectrogram:Tensor = torch.abs(stft)
        phase:Tensor = torch.angle(stft)
        return {"spec":spectrogram, "phase": phase} 

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from TorchDatasetMusDB18 import TorchDatasetMusDB18
    import numpy as np

    data_set = TorchDatasetMusDB18("./dataset/musdb18/train")
    batch_size:int = 1
    data_loader = DataLoader(dataset=data_set,batch_size=batch_size,shuffle=True,num_workers=(batch_size-1),drop_last=True)
    
    process_lr_sr_audio = ProcessLRSRAudio()
    max_value = -np.inf
    min_value = np.inf
    for data in data_loader:
        audio_dict:dict = process_lr_sr_audio.get_lr_hr_dict_keep_sr(data)
        hr_spec = process_lr_sr_audio.get_spectrogram_and_phase_from_audio(audio_dict["hr"])["spec"]
        lr_spec = process_lr_sr_audio.get_spectrogram_and_phase_from_audio(audio_dict["lr"])["spec"]
        max_value = max(max(float(torch.max(hr_spec)),max_value),float(torch.max(lr_spec)))
        min_value = min(min(float(torch.min(hr_spec)),min_value),float(torch.min(lr_spec)))
    
    print(f"max is {max_value} ans min value is {min_value}")