# src/trainers/__init__.py
from .StarLyraMHC_tester import StarLyraMHC_Tester
from .capsnet_tester import Legacytester
from .transpMHC_tester import TranspMHC_Tester



def get_tester(cfg, model_cls):
    if cfg.dataset.train.name == "Anthem_train":
        return Legacytester(cfg, model_cls)
    elif cfg.dataset.train.name == "transpMHC_train":
        return StarLyraMHC_Tester(cfg, model_cls)
    else:
        raise ValueError(f"Unknown trainer type: {cfg.trainer.name}")