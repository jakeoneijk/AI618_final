from collections import OrderedDict
import json
import os
from datetime import datetime
from matplotlib.pyplot import axis
import torch
import torch.utils.data
from tqdm import tqdm
from SpecPlot import SpecPlot
from TorchDatasetMusDB18SpecTest import TorchDatasetMusDB18SpecTest
from core.logger import NoneDict
from model.model import DDPM

class Test:
    def __init__(
        self,
        config_path:str = "./config/musdb.json",
        pretrained_path:str ="./experiments/musdb_220607_163409/checkpoint/I900000_E655",
        sample_rate:int = 16000,
        scale_value:float = 65,
        segment_size:int = 200,
        dataroot:str = "./dataset/musdb18/test",
        data_set_mode:str = "LRHR",
        output_path:str = "./TestOutput/"
        ) -> None:

        self.sample_rate = sample_rate
        self.dataroot = dataroot
        self.data_set_mode = data_set_mode
        self.scale_value = scale_value
        self.config = self.json_parse(config_path)
        self.config['path']['resume_state'] = pretrained_path
        self.output_path = output_path + datetime.now().strftime('%y%m%d-%H%M%S')
        os.makedirs(self.output_path,exist_ok=True)
        self.segment_size = segment_size
        self.spec_plot = SpecPlot(scale_value,output_path=self.output_path)
    
    def json_parse(self,json_path:str):
        json_str = ''
        with open(json_path, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
        opt = json.loads(json_str, object_pairs_hook=OrderedDict)
        return self.dict_to_nonedict(opt)

    def dict_to_nonedict(self,opt):
        if isinstance(opt, dict):
            new_opt = dict()
            for key, sub_opt in opt.items():
                new_opt[key] = self.dict_to_nonedict(sub_opt)
            return NoneDict(**new_opt)
        elif isinstance(opt, list):
            return [self.dict_to_nonedict(sub_opt) for sub_opt in opt]
        else:
            return opt

    def test(self):
        test_set = TorchDatasetMusDB18SpecTest(self.dataroot, self.data_set_mode,self.sample_rate,self.scale_value)
        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        diffusion = DDPM(self.config)
        idx = 0

        for val_data in tqdm(test_data_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()
            visuals["SR"][:,128,:] = 0

            self.spec_plot.model_output_to_spec_db_scale(visuals['SR'],'{}_sr'.format(idx),audio_save=True)
            self.spec_plot.model_output_to_spec_db_scale(visuals['HR'],'{}_hr'.format(idx),audio_save=True)
            self.spec_plot.model_output_to_spec_db_scale(visuals['LR'],'{}_lr'.format(idx),audio_save=True)
                                

if __name__ == "__main__":
    test = Test()
    test.test()
