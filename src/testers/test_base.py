
import logging
from abc import ABC, abstractmethod


class Basetester(ABC):

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)


    @abstractmethod
    def test(self, checkpoints_dir, cfg, data_provider, fold):
        pass
