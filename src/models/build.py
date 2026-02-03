from src.models.registry import MODEL_REGISTRY

def build_model(cfg):
    model_name = cfg.experiment.model

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    return MODEL_REGISTRY[model_name](cfg)
