"""Deterministic, geometry-preserving image corruptions."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from PIL import Image, ImageFilter
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CorruptionKind = Literal[
    "brightness",
    "gaussian_noise",
    "salt_pepper",
    "gaussian_blur",
    "motion_blur",
    "jpeg",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CorruptionLevel(_StrictModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    kind: CorruptionKind
    value: float

    @model_validator(mode="after")
    def validate_value(self) -> CorruptionLevel:
        if not np.isfinite(self.value) or self.value <= 0:
            raise ValueError("corruption value must be finite and positive")
        if self.kind in {"gaussian_noise", "salt_pepper"} and self.value > 1:
            raise ValueError(f"{self.kind} value must not exceed 1")
        if self.kind == "motion_blur" and (
            not self.value.is_integer() or int(self.value) < 3 or int(self.value) % 2 == 0
        ):
            raise ValueError("motion blur kernel must be an odd integer >= 3")
        if self.kind == "jpeg" and (not self.value.is_integer() or self.value > 100):
            raise ValueError("JPEG quality must be an integer in [1, 100]")
        return self


class CorruptionConfig(_StrictModel):
    schema_version: Literal[1]
    seed: int = Field(ge=0, le=2**32 - 1)
    levels: tuple[CorruptionLevel, ...] = Field(min_length=1)

    @field_validator("levels")
    @classmethod
    def unique_level_names(cls, levels: tuple[CorruptionLevel, ...]) -> tuple[CorruptionLevel, ...]:
        names = [level.name for level in levels]
        if len(set(names)) != len(names):
            raise ValueError("corruption level names must be unique")
        return levels


def load_corruptions(path: str | Path) -> CorruptionConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("corruption YAML must contain a mapping")
    return CorruptionConfig.model_validate(payload)


def corruption_fingerprint(config: CorruptionConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _rng(base_seed: int, image_id: str, level_name: str) -> np.random.Generator:
    material = f"{base_seed}\0{image_id}\0{level_name}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(derived_seed)


def _motion_kernel(size: int, angle_degrees: float) -> np.ndarray:
    coordinates = np.arange(size, dtype=np.float32) - (size - 1) / 2
    x, y = np.meshgrid(coordinates, coordinates)
    angle = np.deg2rad(angle_degrees)
    along = x * np.cos(angle) + y * np.sin(angle)
    across = -x * np.sin(angle) + y * np.cos(angle)
    kernel = ((np.abs(along) <= size / 2) & (np.abs(across) <= 0.5)).astype(np.float32)
    kernel[size // 2, size // 2] = 1
    return kernel / kernel.sum()


def _convolve_rgb(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = kernel.shape[0] // 2
    padded = np.pad(array, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape, axis=(0, 1))
    return np.einsum("hwcij,ij->hwc", windows, kernel, optimize=True)


def apply_corruption(
    image: Image.Image,
    level: CorruptionLevel,
    *,
    base_seed: int,
    image_id: str,
) -> Image.Image:
    """Create one corruption from the supplied clean image."""

    clean = image.convert("RGB")
    array = np.asarray(clean, dtype=np.float32)
    rng = _rng(base_seed, image_id, level.name)

    if level.kind == "brightness":
        result = array * level.value
    elif level.kind == "gaussian_noise":
        result = array + rng.normal(0, level.value * 255, size=array.shape)
    elif level.kind == "salt_pepper":
        result = array.copy()
        affected = rng.random(array.shape[:2]) < level.value
        salt = rng.random(array.shape[:2]) < 0.5
        result[affected & salt] = 255
        result[affected & ~salt] = 0
    elif level.kind == "gaussian_blur":
        return clean.filter(ImageFilter.GaussianBlur(radius=level.value))
    elif level.kind == "motion_blur":
        kernel = _motion_kernel(int(level.value), rng.uniform(0, 180))
        result = _convolve_rgb(array, kernel)
    elif level.kind == "jpeg":
        buffer = BytesIO()
        clean.save(
            buffer,
            format="JPEG",
            quality=int(level.value),
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")
    else:
        raise AssertionError(f"unhandled corruption kind: {level.kind}")

    return Image.fromarray(np.clip(np.rint(result), 0, 255).astype(np.uint8), mode="RGB")
