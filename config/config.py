"""
Configuration Module

Copyright (c) 2025 Tekly IT Solutions. All rights reserved.
PROPRIETARY SOFTWARE - Commercial use requires license agreement.
"""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    detection_size: Tuple[int, int] = (160, 160)
    threshold: float = 0.6
    embedding_dim: int = 512  # InsightFace embedding dimension

@dataclass
class CameraConfig:
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30

@dataclass
class DatasetConfig:
    dataset_path: str = "dataset"
    supported_formats: tuple = ('.jpg', '.jpeg', '.png', '.bmp')

@dataclass
class StorageConfig:
    model_path: str = "models/enhanced_face_model.pkl"
    index_path: str = "models/enhanced_face_index.faiss"

model_config = ModelConfig()
camera_config = CameraConfig()
dataset_config = DatasetConfig()
storage_config = StorageConfig()