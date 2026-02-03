# src/trainers/__init__.py
from .capsnet_trainer import CapsNet_Trainer
from .transpMHC_trainer import TranspMHC_Trainer
from .StarLyraMHC_trainer import StarLyraMHC_Trainer

def get_trainer(cfg, model_cls):
    if cfg.dataset.train.name == "Anthem_train":
        return CapsNet_Trainer(cfg, model_cls)
    elif cfg.dataset.train.name == "transpMHC_train":
        return StarLyraMHC_Trainer(cfg, model_cls)
        # return TranspMHC_Trainer(cfg, model_cls)

    else:
        raise ValueError(f"Unknown trainer type: {cfg.trainer.name}")
