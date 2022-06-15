from numpy import ndarray
from torch import Tensor

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

## EDITED by Dongryung Lee adopted from Jaekwon Im's plotting code
class SpecPlot:
    def __init__(self, scale_value:float=65) -> None:
        self.scale_value:float = scale_value
    
    def model_output_to_spec_db_scale(self, model_output:Tensor, save_path:str=""):
        spec:ndarray = model_output.detach().squeeze().numpy() * self.scale_value
        spec_db:ndarray = librosa.amplitude_to_db(spec, ref=np.max)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(spec_db, ax=ax)

        if save_path != "":
            plt.savefig(save_path,dpi=1000)
        
        plt.close()
