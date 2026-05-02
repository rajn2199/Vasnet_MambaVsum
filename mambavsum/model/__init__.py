# model/__init__.py
"""
MambaVSum Model Package

Architecture:
  Input Features -> Multimodal Fusion -> Bidirectional Mamba Encoder
  -> Multi-Scale Temporal Pooling -> Score Regressor -> Frame Scores
"""
from model.mamba import BiMambaBlock, BiMambaEncoder
from model.fusion import MultimodalFusion, GatedFusionUnit
from model.mambavsum import MambaVSum

__all__ = [
    "BiMambaBlock",
    "BiMambaEncoder",
    "MultimodalFusion",
    "GatedFusionUnit",
    "MambaVSum",
]
