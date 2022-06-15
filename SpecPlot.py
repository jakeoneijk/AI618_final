from numpy import ndarray
from torch import Tensor

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


class SpecPlot:
    def __init__(self, scale_value:float=65,sample_rate:int = 16000,output_path:str="./") -> None:
        self.scale_value:float = scale_value
        self.sample_rate = sample_rate
        self.output_path:str = output_path
    
    def model_output_to_spec_db_scale(self, model_output:Tensor, feature_name:str="",audio_save:bool=False):
        spec:ndarray = model_output.detach().squeeze().numpy() * self.scale_value
        spec_db:ndarray = librosa.amplitude_to_db(spec, ref=np.max)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(spec_db, ax=ax)

        if feature_name != "":
            plt.savefig(self.output_path + "/spec_" + feature_name+".png",dpi=1000)
        
        if audio_save:
            print("start griffin lim")
            audio = librosa.griffinlim(spec, hop_length=128)
            sf.write(self.output_path + "/audio_" +feature_name+".wav", audio, self.sample_rate)
        