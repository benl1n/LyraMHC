# src/testers/__init__.py
from .capsnet_tester import Capsnet_tester
from .transpMHC_tester import TranspMHC_Tester
from ..registry import TESTER_REGISTRY


def get_tester(cfg, model_cls):
    tester_name = cfg.dataset.train.name
    tester_cls = TESTER_REGISTRY.get(tester_name)
    return tester_cls(cfg, model_cls)
