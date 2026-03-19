# src/model/__init__.py
from src.registry import MODEL_REGISTRY
from . import LyraMHC

def build_model(cfg):
    model_name = cfg.experiment.model
    model_cls = MODEL_REGISTRY.get(model_name)
    return model_cls(cfg)
