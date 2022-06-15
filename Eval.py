import os

from tqdm import tqdm
import numpy as np
from MetricSound import MetricSound
## EDITED by jaekwon im : wrote whole class
class Eval:
    def __init__(
        self,
        result_data_path:str
        ) -> None:
        
        self.result_data_path = result_data_path
        self.metric = MetricSound()
        self.song_num:int = 50
    
    def eval(self) -> None:
        snr_result_list = list()
        lsd_result_list = list()
        for i in tqdm(range(1,self.song_num + 1)):
            pred_audio_path:str = f"{self.result_data_path}/audio_{i}_sr.wav"
            target_audio_path:str = f"{self.result_data_path}/audio_{i}_hr.wav"
            metric = self.metric.get_metric_from_audio_path(pred_audio_path=pred_audio_path,target_audio_path=target_audio_path)
            snr_result_list.append(metric["snr"])
            lsd_result_list.append(metric["lsd"])
        
        print(f"snr:{np.median(snr_result_list)} , lsd: {np.median(lsd_result_list)}")





if __name__ == "__main__":
    eval = Eval("./TestOutput/220614-182351")
    eval.eval()


