from concurrent.futures import ProcessPoolExecutor
import time
import os
import pickle
import numpy as np
import librosa
from numpy import ndarray
import musdb

class PreprocessorMusDB18:
    def __init__(self,original_data_dir:str,preprocessed_data_dir:str,target_sr:int) -> None:
        self.original_data_dir:str = original_data_dir
        self.preprocessed_data_dir:str = preprocessed_data_dir
        self.subset_split_list:list = [("train","train"),("train","valid"),("test",None)]
        self.mono: bool = True
        self.musdb_origin_sr:int = 44100
        self.target_sr = target_sr

    def preprocess_data(self) -> None:
        meta_param_list:list = self.get_meta_data_param()
        start_time:float = time.time()

        with ProcessPoolExecutor(max_workers=None) as pool:
            pool.map(self.preprocess_one_data, meta_param_list)

        print("Finish preprocess. {:.3f} s".format(time.time() - start_time))
    
    def get_meta_data_param(self) -> list:
        param_list:list = []
        for subset_split in self.subset_split_list:
            mus = musdb.DB(root=self.original_data_dir, subsets=subset_split[0], split=subset_split[1])
            print("Subset: {}, Split: {}, Total pieces: {}".format(subset_split[0], subset_split[1], len(mus)))
            for track_index in range(len(mus.tracks)):
                param = (subset_split,track_index)
                param_list.append(param)
        return param_list
    
    def get_audio_from_musdb_track(self, musdb_track, mono:bool) -> ndarray:
        audio:ndarray = musdb_track.audio.T

        if mono:
            audio = np.mean(audio, axis=0)
            
        resampled_audio:ndarray = librosa.core.resample(audio, orig_sr=self.musdb_origin_sr, target_sr=self.target_sr, res_type="kaiser_fast")

        return resampled_audio
    
    def preprocess_one_data(self,param: tuple) -> None:
        (subset_split, track_index) = param
        mus = musdb.DB(root=self.original_data_dir, subsets=subset_split[0], split=subset_split[1])
        track = mus.tracks[track_index]

        feature_dict:dict = dict()
        feature_dict["audio"] = self.get_audio_from_musdb_track(track,self.mono)
        datatype:str = subset_split[1] if subset_split[1] is not None else subset_split[0]
        
        save_path:str = f"{self.preprocessed_data_dir}/{datatype}"
        os.makedirs(save_path,exist_ok=True)

        with open(save_path + f"/{track.name}.pkl",'wb') as file_writer:
            pickle.dump(feature_dict,file_writer)

        print("{} Write to {}".format(track_index, save_path))
        
        
    
if __name__ == "__main__":
    preprocessor_musdb_18 = PreprocessorMusDB18("../data/musdb18","../data/musdb18_processed",16000)
    preprocessor_musdb_18.preprocess_data()