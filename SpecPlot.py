from numpy import ndarray
from torch import Tensor

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

class SpecPlot:
    def __init__(self, sr:int=16000) -> None:
        self.sr = sr
    
    def model_output_to_spec_db_scale(self, model_output:Tensor, save_path:str=""):
        audio:ndarray = model_output.detach().squeeze().numpy()

        # Save audio
        sf.write(f"{save_path}.wav", audio, self.sr)

        # Save stft
        n_fft = 2048
        stft = librosa.stft(audio, window='hann', n_fft=n_fft, hop_length=n_fft//4)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft), ref=np.max), ax=ax)
        plt.savefig(f"{save_path}.png",dpi=1000)
        