from abc import ABC, abstractmethod
from tqdm import tqdm

from src.callbacks import ModelCheckPointCallBack, EarlyStopCallBack

class BaseTrainer(ABC):

    def __init__(self, model, config, device):
        pass
    @abstractmethod
    def train(self, checkpoints_dir, train_loader, val_loader, epochs):
        pass

