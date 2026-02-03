# src/models/registry.py
from src.models.components.LyraMHC import LyraMHC
from src.models.components.BiLyraMHC import BiLyraMHC
from src.models.components.StarLyraMHC import StarLyraMHC

MODEL_REGISTRY = {
    "LyraMHC": LyraMHC,
    "StarLyraMHC": StarLyraMHC,
    "BiLyraMHC": BiLyraMHC

}