from numpy import ndarray
from torch import Tensor

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

class SpecPlot:
    def __init__(self, sr:int=16000, scale_value:float=65) -> None:
        self.sr = sr
        self.scale_value:float = scale_value
    
    def model_output_to_spec_db_scale(self, model_output:Tensor, save_path:str=""):
        spec:ndarray = model_output.detach().squeeze().numpy() * self.scale_value
        spec_db:ndarray = librosa.amplitude_to_db(spec, ref=np.max)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(spec_db, ax=ax)
        plt.savefig(f"{save_path}.png",dpi=1000)

        audio = librosa.griffinlim(spec)
        sf.write(f"{save_path}.wav", audio, self.sr)
        