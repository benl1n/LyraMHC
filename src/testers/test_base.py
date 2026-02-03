import torch
import torch.nn as nn
import os
import logging
from abc import ABC, abstractmethod  # 引入抽象基类
from tqdm import tqdm

from src.callbacks import ModelCheckPointCallBack, EarlyStopCallBack

class Basetester(ABC):
    """
    基类：负责通用的管家工作（设备管理、优化器初始化、Epoch循环、日志、Checkpoint）
    """

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

        #
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=config.start_lr,
        #     weight_decay=1e-6
        # )
        #
        # # 3. 初始化调度器
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer,
        #     threshold=config.loss_delta,
        #     patience=4,
        #     min_lr=config.min_lr,
        #     factor=0.2
        # )
        #
        # # 4. 初始化 Callbacks
        # self.early_stop = EarlyStopCallBack(patience=10, delta=config.loss_delta)
        # # Checkpoint 需要知道具体的保存路径，这里留个空或者通用命名

    @abstractmethod
    def test(self, checkpoints_dir, cfg, data_provider, fold):
        pass
