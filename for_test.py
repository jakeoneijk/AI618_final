from lightning_model import NuWave
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
import torch
from tqdm import tqdm
from scipy.io.wavfile import write as swrite
import pdb
from MetricSound import MetricSound
import numpy as np


def test(args):
    hparams = OC.load('hparameter.yaml')
    hparams.save = args.save or False
    model = NuWave(hparams, False).cuda()
    
    metric = MetricSound()
    
    if args.ema:
        ckpt_path = glob(os.path.join(hparams.log.checkpoint_dir,
                         f'*_epoch={args.resume_from}_EMA'))[-1]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)
                         
    else:
        #ckpt_path = glob(os.path.join(hparams.log.checkpoint_dir, f'*_epoch={args.resume_from}.ckpt'))[-1]
        ckpt_path = os.path.join(hparams.log.checkpoint_dir, 'last.ckpt')
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
    print(ckpt_path)
    model.eval()
    model.freeze()
    os.makedirs(hparams.log.test_result_dir, exist_ok=True)

    results=[]
    for i in range(1): # 5
        snr=[]
        base_snr=[]
        lsd=[]
        base_lsd=[]
        SNR_list = []
        LDS_list = [] 
        t = model.test_dataloader()
        print(f'{len(t)} test data')
        
        for j, batch in tqdm(enumerate(t)):
            wav, wav_l = batch
            wav=wav.cuda()
            wav_l = wav_l.cuda()
            
            _, L = wav.shape 
            #pdb.set_trace()
            '''
            step_size = 32768*8
            if L < step_size : 
                pdb.set_trace()
            
            wav_ups = [] 
            for start in range(0, L, step_size) : 
                inp = wav_l[:,start: start+step_size] 
                oup = model.sample(inp, model.hparams.ddpm.infer_step)
                wav_ups.append(oup)
            
            
            wav_up = torch.cat(wav_ups, dim=1)
            
            if not wav_up.shape == wav.shape : 
                pdb.set_trace()'''

            sr = int(hparams.audio.sr)
            if L < sr*63 : 
                pdb.set_trace()
            wav_l = wav_l[:,sr*60:sr*63]            
            wav = wav[:,sr*60:sr*63]   
            wav_up = model.sample(wav_l, model.hparams.ddpm.infer_step)
            
            
            pred = wav_up.cpu().detach().numpy()
            target = wav.cpu().detach().numpy()
            SNR = metric.signal_to_noise(pred, target)
            LDS = metric.lds_log_spectral_distance(np.squeeze(pred), np.squeeze(target))
            SNR_list.append(SNR)
            LDS_list.append(LDS)
            
            
            
            snr.append(model.snr(wav_up,wav).detach().cpu())
            base_snr.append(model.snr(wav_l, wav).detach().cpu())
            lsd.append(model.lsd(wav_up,wav).detach().cpu())
            base_lsd.append(model.lsd(wav_l, wav).detach().cpu())
            if args.save and i==0:
                swrite(f'{hparams.log.test_result_dir}/test_{j}_up.wav',
                       hparams.audio.sr, wav_up[0].detach().cpu().numpy())
                swrite(f'{hparams.log.test_result_dir}/test_{j}_orig.wav',
                       hparams.audio.sr, wav[0].detach().cpu().numpy())
                swrite(f'{hparams.log.test_result_dir}/test_{j}_linear.wav',
                       hparams.audio.sr, wav_l[0].detach().cpu().numpy())
                swrite(f'{hparams.log.test_result_dir}/test_{j}_down.wav',
                       hparams.audio.sr//hparams.audio.ratio, wav_l[0,::hparams.audio.ratio].detach().cpu().numpy())


        SNR = np.array(SNR_list)
        LDS = np.array(LDS_list)
        print(np.median(SNR))
        print(np.mean(SNR))
        print(np.median(LDS))
        print(np.mean(LDS))       

        pdb.set_trace()        
        
        snr = torch.stack(snr, dim =0).mean()
        base_snr = torch.stack(base_snr, dim =0).mean()
        lsd = torch.stack(lsd, dim =0).mean()
        base_lsd = torch.stack(base_lsd, dim =0).mean()
        dict = {
            'snr': snr.item(),
            'base_snr': base_snr.item(),
            'lsd': lsd.item(),
            'base_lsd': base_lsd.item(),
        }
        results.append(torch.stack([snr, base_snr, lsd, base_lsd],dim=0).unsqueeze(-1))
        print(dict)
        
        
    results = torch.cat(results,dim=1)
    for i in range(4):
        print(torch.mean(results[i]),torch.std(results[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume_from', type =int,\
            required = False, help = "Resume Checkpoint epoch number")
    parser.add_argument('-e', '--ema', action = "store_true",\
            required = False, help = "Start from ema checkpoint")
    parser.add_argument('--save', action = "store_true",\
            required = False, help = "Save file")

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False
    test(args)
