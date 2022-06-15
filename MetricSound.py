from numpy import ndarray

import numpy as np
import librosa

class MetricSound:
    
    def signal_to_noise(self,pred_waveform:ndarray,target_waveform:ndarray) -> float:
        return 10.*np.log10(np.sqrt(np.sum(target_waveform**2))/np.sqrt(np.sum((target_waveform - pred_waveform)**2)))
    
    def lds_log_spectral_distance(self,pred_waveform:ndarray,target_waveform:ndarray, nfft:int = 256) -> float:
        pred_spectrogram:ndarray = librosa.core.stft(pred_waveform, n_fft=nfft)
        target_spectrogram:ndarray = librosa.core.stft(target_waveform, n_fft=nfft)

        pred_log_spectral_power_magnitude:ndarray = np.log10(np.abs(pred_spectrogram)**2)
        target_log_spectral_power_magnitude:ndarray = np.log10(np.abs(target_spectrogram)**2)
        
        diff_squared:ndarray = (target_log_spectral_power_magnitude - pred_log_spectral_power_magnitude)**2
        frequency_axis:int = 0

        return np.mean(np.sqrt(np.mean(diff_squared, axis=frequency_axis)))
    
    def get_metric_from_audio_path(self,pred_audio_path,target_audio_path):
        pred_audio,sr = librosa.load(pred_audio_path,sr=None)
        target_audio,sr = librosa.load(target_audio_path,sr=None)
        snr = self.signal_to_noise(pred_audio,target_audio)
        lsd = self.lds_log_spectral_distance(pred_audio,target_audio)
        return {"snr":snr, "lsd":lsd}