import sys
import datetime
import os
import pandas as pd

import torch
from hydra.core.hydra_config import HydraConfig
from torch.optim import lr_scheduler
from torch import optim
import torch.nn as nn


#############################################################################################
#
# Test
#
#############################################################################################
from src.data_provider.capsnet_data_provider import DataProvider
from src.encode import ENCODING_METHOD_MAP
from src.logger import setup_logging, log_to_file
from src.result_writer import weeekly_result_writer
from src.testers.test_base import Basetester
from src.utlis import get_data, get_model_save_path


class Legacytester(Basetester):

    def __init__(self, cfg, model):

        self.config = cfg

        self.model = model


        super().__init__(model=model, config=cfg, device=cfg.train.device)

    def batch_test(self, model, device, data):
        hla_a, hla_mask, hla_a2, hla_mask2, pep, pep_mask, pep2, pep_mask2, uid_list = data
        pred_ic50 = model(hla_a.to(device), pep.to(device))

        return pred_ic50, uid_list





    def test(self, checkpoints_dir, cfg, data_provider, fold):
        # if not config.do_test:
        #     log_to_file('Skip testing', 'Not enabled testing')
        #     return
        device = self.config.train.device

        temp_list = []
        for p in range(cfg.dataset.params.base_model_count):
            for k in range(cfg.dataset.params.model_count):
                # load and prepare model
                save_path_best = get_model_save_path(cfg, checkpoints_dir, k, cfg.dataset.train.name, prefix="best")
                state_dict = torch.load(save_path_best)
                model = self.model
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()
                temp_dict = {}
                data_provider.new_epoch()

                for _ in range(data_provider.test_steps()):
                    data = data_provider.batch_test()

                    with torch.no_grad():
                        pred_ic50, uid_list = self.batch_test(model, device, data)
                        # print(pred_ic50)
                        for i, uid in enumerate(uid_list):
                            temp_dict[uid] = pred_ic50[i].item()
                temp_list.append(temp_dict)

        # average score of the emsemble model
        result_dict = temp_list[0]
        if cfg.dataset.params.model_count > 1:
            for k in range(1, cfg.dataset.params.model_count):
                for j in result_dict.keys():
                    result_dict[j] += temp_list[k][j]

        if cfg.dataset.params.base_model_count > 1:
            for p in range(1, cfg.dataset.params.base_model_count):
                for k in range(cfg.dataset.params.model_count):
                    for j in result_dict.keys():
                        result_dict[j] += temp_list[p * cfg.dataset.params.model_count + k][j]

        for j in result_dict.keys():
            result_dict[j] = result_dict[j] / (cfg.dataset.params.model_count * cfg.dataset.params.base_model_count)

        # print(result_dict)
        result_file = weeekly_result_writer(result_dict, cfg)
        log_to_file('Testing result file', result_file)


#############################################################################################
#
# Main
#
#############################################################################################

    def fit(self):
        # parse config
        cfg = self.config

        # encoding func
        encoding_func = ENCODING_METHOD_MAP[cfg.train.encoding_method]
        encoding_func2 = ENCODING_METHOD_MAP[cfg.train.encoding_method2]

        train_file, test_file, _ = get_data(cfg.train.name,
                                cfg.train.task,
                                cfg.train.data_path)

        data_provider = []
        for p in range(cfg.dataset.params.base_model_count):
            temp_provider = DataProvider(
                train_file,
                test_file,
                cfg.train.name,
                cfg.train.task,
                cfg.train.data_path,
                encoding_func,
                encoding_func2,
                cfg.dataset.train.batch_size,
                cfg.dataset.params.max_len_hla,
                cfg.dataset.params.max_len_pep,
                cfg.dataset.params.model_count,
            )
            data_provider.append(temp_provider)

        log_to_file('max_len_hla', data_provider[0].max_len_hla)
        log_to_file('max_len_pep', data_provider[0].max_len_pep)

        checkpoints_path = setup_logging(HydraConfig.get().runtime.output_dir)
        self.test(checkpoints_path, cfg, data_provider[0], fold=None)
