## EDITED by Dongryung Lee for inpainting

from pyrsistent import v
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import pdb
from SpecPlot import SpecPlot
import librosa
import soundfile as sf
from TorchDatasetMusDB18Spec import TorchDatasetMusDB18Spec
from MetricSound import MetricSound
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_musdb.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            #train_set = Data.create_dataset(dataset_opt, phase)
            train_set = TorchDatasetMusDB18Spec(dataset_opt['dataroot'], dataset_opt['mode'],
                                            dataset_opt['sr'], dataset_opt['segment_length_second'], 
                                            dataset_opt['samples_per_track'], dataset_opt["max_value_of_spec"], 
                                            need_mask=True, need_audio=True)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            #val_set = Data.create_dataset(dataset_opt, phase)
            val_set = TorchDatasetMusDB18Spec(dataset_opt['dataroot'], dataset_opt['mode'],
                                            dataset_opt['sr'], dataset_opt['segment_length_second'],
                                            dataset_opt['samples_per_track'], dataset_opt["max_value_of_spec"],
                                            need_mask=True, need_audio=True)            
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    spec_plot = SpecPlot(opt['datasets']["train"]["max_value_of_spec"])

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch

    logger.info('Begin Model Evaluation.')
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    # metric
    metric = MetricSound()
    result = np.zeros((len(val_loader.dataset),2))
    for _,  val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test_inpaint(continuous=False)
        visuals = diffusion.get_current_visuals()

        visuals['SR'][:,128,:0]
        lr_inv = librosa.griffinlim(visuals['LR'].detach().cpu().numpy()[0], hop_length=128)
        #hr_inv = librosa.griffinlim(visuals['HR'].detach().cpu().numpy()[0], hop_length=128)
        hr_inv = val_data['audio'][0].cpu().numpy()
        sr_inv = librosa.griffinlim(visuals['SR'].detach().cpu().numpy(), hop_length=128)

        sf.write('{}/{}_{}_lr.wav'.format(result_path, current_step, idx), lr_inv.T, 16000)
        sf.write('{}/{}_{}_hr.wav'.format(result_path, current_step, idx), hr_inv.T, 16000)
        sf.write('{}/{}_{}_sr.wav'.format(result_path, current_step, idx), sr_inv.T, 16000)
        
        spec_plot.model_output_to_spec_db_scale(visuals['SR'],'{}/{}_{}_sr.png'.format(result_path, current_step, idx))
        spec_plot.model_output_to_spec_db_scale(visuals['HR'],'{}/{}_{}_hr.png'.format(result_path, current_step, idx))
        spec_plot.model_output_to_spec_db_scale(visuals['LR'],'{}/{}_{}_lr.png'.format(result_path, current_step, idx))
        spec_plot.model_output_to_spec_db_scale(visuals['INF'],'{}/{}_{}_inf.png'.format(result_path, current_step, idx))

        snr = metric.signal_to_noise(sr_inv, hr_inv)
        lsd = metric.lds_log_spectral_distance(sr_inv, hr_inv)

        result[idx-1, 0] = snr.item()
        result[idx-1, 1] = lsd.item()

        # save metrics
        np.save(result_path + '/metrics.npy', result, allow_pickle=False)
        print('SNR: {}, LSD:{}'.format(np.median(result[:idx, 0]), np.median(result[:idx, 1])))