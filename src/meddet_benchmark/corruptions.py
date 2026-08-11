"""Deterministic, geometry-preserving Albumentations corruption pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A

CorruptionKind = Literal[
    "brightness",
    "gaussian_noise",
    "salt_pepper",
    "gaussian_blur",
    "motion_blur",
    "jpeg",
]
CorruptionFamily = Literal["lighting", "noise", "blur", "compression"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SeverityLevel(_StrictModel):
    """One ordered corruption strength read from YAML."""

    severity: int = Field(ge=1, le=5)
    value: float

    @field_validator("value")
    @classmethod
    def finite_positive_value(cls, value: float) -> float:
        if not np.isfinite(value) or value <= 0:
            raise ValueError("corruption value must be finite and positive")
        return value


class CorruptionDefinition(_StrictModel):
    """One corruption type and its complete ordered severity curve."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    family: CorruptionFamily
    kind: CorruptionKind
    unit: str = Field(min_length=1)
    levels: tuple[SeverityLevel, ...] = Field(min_length=3, max_length=5)

    @model_validator(mode="after")
    def validate_curve(self) -> CorruptionDefinition:
        severities = [level.severity for level in self.levels]
        if severities != list(range(1, len(self.levels) + 1)):
            raise ValueError("severity levels must be contiguous and ordered from 1")
        values = [level.value for level in self.levels]
        if self.kind in {"gaussian_noise", "salt_pepper"} and any(
            value > 1 for value in values
        ):
            raise ValueError(f"{self.kind} values must not exceed 1")
        if self.kind == "motion_blur" and any(
            not value.is_integer() or int(value) < 3 or int(value) % 2 == 0
            for value in values
        ):
            raise ValueError("motion-blur kernels must be odd integers >= 3")
        if self.kind == "jpeg" and any(
            not value.is_integer() or value > 100 for value in values
        ):
            raise ValueError("JPEG qualities must be integers in [1, 100]")
        if self.kind == "brightness":
            all_dark = all(value < 1 for value in values)
            all_bright = all(value > 1 for value in values)
            if not (all_dark or all_bright):
                raise ValueError("a brightness curve must remain on one side of 1.0")
            expected = sorted(values, reverse=all_dark)
        elif self.kind == "jpeg":
            expected = sorted(values, reverse=True)
        else:
            expected = sorted(values)
        if values != expected:
            raise ValueError("values must progress from mild to severe")
        return self


class CorruptionConfig(_StrictModel):
    """The corruption-only portion of the Phase 6 experiment config."""

    schema_version: Literal[2]
    seed: int = Field(ge=0, le=2**32 - 1)
    corruptions: tuple[CorruptionDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_matrix(self) -> CorruptionConfig:
        names = [item.name for item in self.corruptions]
        if len(set(names)) != len(names):
            raise ValueError("corruption names must be unique")
        return self


class CorruptionCondition(_StrictModel):
    """A single expanded corruption/severity evaluation condition."""

    name: str
    family: CorruptionFamily
    kind: CorruptionKind
    unit: str
    severity: int
    value: float

    @property
    def condition_id(self) -> str:
        return f"{self.name}_s{self.severity}"


def load_corruptions(path: str | Path) -> CorruptionConfig:
    """Load the corruption matrix from the full Phase 6 YAML."""

    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("corruption YAML must contain a mapping")
    subset = {
        "schema_version": payload.get("schema_version"),
        "seed": payload.get("seed"),
        "corruptions": payload.get("corruptions"),
    }
    return CorruptionConfig.model_validate(subset)


def expand_conditions(config: CorruptionConfig) -> tuple[CorruptionCondition, ...]:
    """Expand every type by every configured severity in stable YAML order."""

    return tuple(
        CorruptionCondition(
            name=definition.name,
            family=definition.family,
            kind=definition.kind,
            unit=definition.unit,
            severity=level.severity,
            value=level.value,
        )
        for definition in config.corruptions
        for level in definition.levels
    )


def corruption_fingerprint(config: CorruptionConfig) -> str:
    """Hash the exact ordered corruption matrix and its stochastic seed."""

    encoded = json.dumps(
        config.model_dump(mode="json"),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _derived_seed(base_seed: int, image_id: str, condition_id: str) -> int:
    material = f"{base_seed}\0{image_id}\0{condition_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little")


def _gaussian_kernel_size(sigma: float) -> int:
    size = max(3, int(np.ceil(6 * sigma + 1)))
    return size if size % 2 else size + 1


def build_transform(condition: CorruptionCondition) -> A.Compose:
    """Build one always-on Albumentations transform from config values."""

    value = condition.value
    if condition.kind == "brightness":
        transform = A.MultiplicativeNoise(
            multiplier=(value, value),
            per_channel=False,
            elementwise=False,
            p=1,
        )
    elif condition.kind == "gaussian_noise":
        transform = A.GaussNoise(
            std_range=(value, value),
            mean_range=(0.0, 0.0),
            per_channel=False,
            p=1,
        )
    elif condition.kind == "salt_pepper":
        transform = A.SaltAndPepper(
            amount=(value, value),
            salt_vs_pepper=(0.5, 0.5),
            p=1,
        )
    elif condition.kind == "gaussian_blur":
        kernel = _gaussian_kernel_size(value)
        transform = A.GaussianBlur(
            blur_limit=(kernel, kernel),
            sigma_limit=(value, value),
            p=1,
        )
    elif condition.kind == "motion_blur":
        kernel = int(value)
        transform = A.MotionBlur(
            blur_limit=(kernel, kernel),
            allow_shifted=False,
            angle_range=(0.0, 360.0),
            direction_range=(0.0, 0.0),
            p=1,
        )
    elif condition.kind == "jpeg":
        quality = int(value)
        transform = A.ImageCompression(
            compression_type="jpeg",
            quality_range=(quality, quality),
            p=1,
        )
    else:
        raise AssertionError(f"unhandled corruption kind: {condition.kind}")
    return A.Compose([transform], p=1)


class CorruptionApplier:
    """Reusable deterministic transform for one grid condition."""

    def __init__(self, condition: CorruptionCondition, *, base_seed: int) -> None:
        self.condition = condition
        self.base_seed = base_seed
        self.transform = build_transform(condition)

    def __call__(self, image: Image.Image, *, image_id: str) -> Image.Image:
        clean = np.asarray(image.convert("RGB"), dtype=np.uint8)
        self.transform.set_random_seed(
            _derived_seed(self.base_seed, image_id, self.condition.condition_id)
        )
        result = self.transform(image=clean)["image"]
        if result.shape != clean.shape or result.dtype != np.uint8:
            raise ValueError("corruption changed image geometry or uint8 dtype")
        return Image.fromarray(result, mode="RGB")


def apply_corruption(
    image: Image.Image,
    condition: CorruptionCondition,
    *,
    base_seed: int,
    image_id: str,
) -> Image.Image:
    """Apply one deterministic corruption without mutating the input image."""

    return CorruptionApplier(condition, base_seed=base_seed)(image, image_id=image_id)
