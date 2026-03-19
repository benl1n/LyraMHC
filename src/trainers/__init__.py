# src/trainers/__init__.py
from .capsnet_trainer import CapsNet_Trainer
from src.registry import TRAINER_REGISTRY
from .transpMHC_trainer import ModernTrainer


def get_trainer(cfg, model_cls):
    trainer_name = cfg.dataset.train.name
    trainer_cls = TRAINER_REGISTRY.get(trainer_name)
    return trainer_cls(cfg, model_cls)