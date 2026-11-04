"""Deterministic lightweight features and an offline DINOv2 adapter."""

import gc
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.config import ColorConfig, EmbeddingConfig
from app.features.color import extract_color


class Embedder(Protocol):
    """Embed image paths into rows of unit vectors."""

    def embed(self, paths: list[Path]) -> NDArray[np.float32]: ...

    def close(self) -> None: ...


class DummyEmbedder:
    """CNN-free colour distribution, spatial colour and contrast descriptor."""

    def __init__(self, embedding: EmbeddingConfig, color: ColorConfig) -> None:
        self.grid = embedding.dummy_grid
        self.color = color
        self.dimension = int(np.prod(color.hsv_bins)) + self.grid * self.grid * 3 + 3

    def embed(self, paths: list[Path]) -> NDArray[np.float32]:
        """Compute deterministic unit-normalized descriptors from local JPEGs."""
        vectors: list[NDArray[np.float32]] = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot decode crop {path}")
            hist, _ = extract_color(image, self.color)
            spatial = cv2.resize(image, (self.grid, self.grid), interpolation=cv2.INTER_AREA) / 255
            contrast = image.std(axis=(0, 1)) / 255
            vector = np.concatenate([np.sqrt(hist), spatial.ravel(), contrast]).astype(np.float32)
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.stack(vectors) if vectors else np.empty((0, self.dimension), dtype=np.float32)

    def close(self) -> None:
        """No model resources are retained by the dummy backend."""


class DinoEmbedder:
    """Batch DINOv2 CLS + mean-patch tokens, with no runtime network requests."""

    def __init__(self, model_path: Path, batch_size: int, device: str = "auto") -> None:
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"DINOv2 must be provisioned locally first: {model_path}")
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch: Any = torch
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_path), local_files_only=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        self.model.eval().to(self.device)
        self.dimension = self.model.config.hidden_size * 2
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    def embed(self, paths: list[Path]) -> NDArray[np.float32]:
        """Embed crop batches; CUDA uses FP16 autocast and sequential execution."""
        output: list[NDArray[np.float32]] = []
        torch = self.torch
        for start in range(0, len(paths), self.batch_size):
            images: list[Image.Image] = []
            for path in paths[start : start + self.batch_size]:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            context = (
                torch.autocast("cuda", dtype=torch.float16)
                if self.device.startswith("cuda")
                else nullcontext()
            )
            with torch.inference_mode(), context:
                tokens = self.model(**inputs).last_hidden_state
                vector = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)
                vector = torch.nn.functional.normalize(vector.float(), dim=1)
            output.append(vector.cpu().numpy())
            for image in images:
                image.close()
        return np.concatenate(output) if output else np.empty((0, self.dimension), dtype=np.float32)

    def peak_memory_bytes(self) -> int:
        """Report peak CUDA allocated memory (zero for CPU execution)."""
        return int(self.torch.cuda.max_memory_allocated()) if self.device.startswith("cuda") else 0

    def close(self) -> None:
        """Release model memory before another heavy stage starts."""
        self.model = None
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()
