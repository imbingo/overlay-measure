from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
import hashlib
import numpy as np
from ..models import DetectionParams, ImageData, MeasurementConfig, Roi
from ..image_loader import load_image
from ..measurement_units import points_px_to_um


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return (
        points @ np.asarray(matrix, dtype=np.float64)[:2, :2].T
        + np.asarray(matrix)[:2, 2]
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Calibration:
    x: float = 0.1
    y: float = 0.1

    def __post_init__(self):
        if not np.isfinite([self.x, self.y]).all() or min(self.x, self.y) <= 0:
            raise ValueError("X/Y 像素尺寸必须为有限正数")

    @property
    def matrix(self):
        return np.diag([self.x, self.y, 1.0])

    def config(self):
        return MeasurementConfig(pixel_size_x_um=self.x, pixel_size_y_um=self.y)

    def physical(self, points):
        return points_px_to_um(points, self.config())


def clockwise_matrix(shape, turns):
    """Native -> np.rot90(image, -turns), pixel centers; no interpolation."""
    h, w = shape
    matrix = np.eye(3)
    for _ in range(turns % 4):
        matrix = (
            np.array([[0.0, -1.0, h - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ matrix
        )
        h, w = w, h
    return matrix


@dataclass
class DiagnosticImage:
    original: ImageData
    calibration: Calibration = field(default_factory=Calibration)
    turns: int = 0
    sha256: str = ""
    _cache: object = field(default=None, init=False, repr=False)

    @classmethod
    def load(cls, path, calibration=None):
        # Hash the same byte snapshot decoded by load_image; reject concurrent changes.
        digest = file_sha256(path)
        image = load_image(str(path))
        if file_sha256(path) != digest:
            raise ValueError("图像在读取期间发生变化，请重新导入")
        image.gray.setflags(write=False)
        return cls(image, calibration or Calibration(), sha256=digest)

    @property
    def oriented(self):
        if self.turns % 4 == 0:
            return self.original
        key = (id(self.original.gray), self.turns % 4)
        if self._cache is None or self._cache[0] != key:
            gray = np.ascontiguousarray(np.rot90(self.original.gray, -self.turns % 4))
            gray.setflags(write=False)
            self._cache = (key, replace(self.original, gray=gray))
        return self._cache[1]

    @property
    def oriented_calibration(self):
        return (
            Calibration(self.calibration.y, self.calibration.x)
            if self.turns % 2
            else self.calibration
        )

    @property
    def native_to_oriented(self):
        return clockwise_matrix(self.original.gray.shape, self.turns)


@dataclass
class Registration:
    matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3)
    )  # B oriented um -> A um
    method: str = "unregistered"
    features: str = ""
    residual_um: float | None = None
    response: float | None = None
    reliable: bool = False
    warning: str = "尚未配准"
    points_a_px: list = field(default_factory=list)
    points_b_px: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def rotation_deg(self):
        return float(np.degrees(np.arctan2(self.matrix[1, 0], self.matrix[0, 0])))


@dataclass
class DiagnosticsSession:
    a: DiagnosticImage | None = None
    b: DiagnosticImage | None = None
    params: DetectionParams = field(default_factory=DetectionParams)
    roi: Roi | None = None
    seed_b_roi: Roi | None = None
    registration: Registration = field(default_factory=Registration)
    result: object = None

    @classmethod
    def from_parameters(cls, params, config):
        # No mutable references to production recipe/config/ROI.
        return cls(params=deepcopy(params))

    def invalidate(self, registration=False):
        self.result = None
        if registration:
            self.registration = Registration()

    def require_pair(self):
        if self.a is None or self.b is None:
            raise ValueError("请导入 Image A 和 Image B")

    def sampling_b(self):
        self.require_pair()
        matrix = np.asarray(self.registration.matrix, np.float64)
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-10)
            or not np.allclose(matrix[:2, :2].T @ matrix[:2, :2], np.eye(2), atol=1e-8)
            or not np.isclose(np.linalg.det(matrix[:2, :2]), 1.0, atol=1e-8)
        ):
            raise ValueError("物理配准仅允许平移和刚性旋转，禁止仿射缩放或非刚性变形")
        return (
            np.linalg.inv(self.b.oriented_calibration.matrix)
            @ np.linalg.inv(matrix)
            @ self.a.oriented_calibration.matrix
        )
