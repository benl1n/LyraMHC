from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    def __init__(self, model, config, device):
        pass

    @abstractmethod
    def train(self, checkpoints_dir, train_loader, val_loader, epochs):
        pass
