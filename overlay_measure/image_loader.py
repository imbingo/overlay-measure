from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import ImageData


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".csv", ".txt"}


def _load_raster_image_unicode_safe(path: str) -> np.ndarray:
    """Load raster image with Windows/Chinese-path compatibility.

    cv2.imread() often returns None on Windows when the file path contains
    Chinese characters or other non-ASCII characters. Reading bytes with
    np.fromfile() and decoding with cv2.imdecode() avoids that issue.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"文件不存在：{path}")
    if not p.is_file():
        raise ValueError(f"不是有效文件：{path}")

    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"无法读取文件字节流：{path}\n原因：{exc}") from exc

    if data.size == 0:
        raise ValueError(f"文件为空：{path}")

    arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(
            f"无法解码图像：{path}\n"
            "可能原因：文件损坏、扩展名与实际格式不一致，或该图像编码不受 OpenCV 支持。"
        )
    return arr


def _load_numeric_matrix(path: str, ext: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"文件不存在：{path}")
    delimiter = "," if ext == ".csv" else None
    try:
        return np.loadtxt(str(p), delimiter=delimiter)
    except UnicodeDecodeError:
        # Some instruments export csv/txt using GBK/ANSI on Windows.
        return np.loadtxt(str(p), delimiter=delimiter, encoding="gbk")
    except Exception as exc:
        raise ValueError(f"无法读取矩阵文件：{path}\n原因：{exc}") from exc


def load_image(path: str) -> ImageData:
    """Load image-like data as a 2D float32 grayscale array.

    Supported:
      - png/jpg/jpeg/bmp/tif/tiff through unicode-safe OpenCV decoding
      - npy as 2D array
      - csv/txt as numeric matrix

    .sur is intentionally not handled here because vendor implementations differ.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        if ext == ".sur":
            raise ValueError(
                ".sur 文件格式没有内置通用解析器。请先导出 tif/csv，或提供样例 .sur 后定制解析。"
            )
        raise ValueError(f"不支持的文件格式：{ext}")

    if ext == ".npy":
        if not p.exists():
            raise ValueError(f"文件不存在：{path}")
        try:
            arr = np.load(str(p))
        except Exception as exc:
            raise ValueError(f"无法读取 npy 文件：{path}\n原因：{exc}") from exc
    elif ext in {".csv", ".txt"}:
        arr = _load_numeric_matrix(str(p), ext)
    else:
        arr = _load_raster_image_unicode_safe(str(p))
        if arr.ndim == 3:
            # cv2 decodes as BGR/BGRA. Convert to grayscale.
            if arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
            elif arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"不支持的图像通道数：{arr.shape[2]}")

    arr = np.asarray(arr)
    source_dtype = str(arr.dtype)
    source_finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else np.asarray([], dtype=float)
    source_min = float(np.min(source_finite)) if source_finite.size else 0.0
    source_max = float(np.max(source_finite)) if source_finite.size else 0.0
    if arr.ndim != 2:
        raise ValueError(f"图像/矩阵必须是二维数据，当前维度：{arr.shape}")

    if not np.isfinite(arr).all():
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            arr = np.nan_to_num(
                arr,
                nan=float(np.nanmedian(finite)),
                posinf=float(np.nanmax(finite)),
                neginf=float(np.nanmin(finite)),
            )

    gray = arr.astype(np.float32)
    return ImageData(
        path=str(p),
        gray=gray,
        display_name=p.name,
        source_dtype=source_dtype,
        source_min=source_min,
        source_max=source_max,
    )


def display_to_uint8(image: ImageData, enhanced: bool = False) -> np.ndarray:
    """Convert image data for display without changing measurement pixels."""
    gray = np.asarray(image.gray, dtype=np.float32)
    if enhanced:
        return normalize_to_uint8(gray)

    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    try:
        dtype = np.dtype(image.source_dtype) if image.source_dtype else None
    except TypeError:
        dtype = None
    if dtype is not None and np.issubdtype(dtype, np.bool_):
        return (gray > 0).astype(np.uint8) * 255
    if dtype is not None and np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        lo, hi = float(info.min), float(info.max)
    else:
        lo, hi = float(image.source_min), float(image.source_max)
        if lo >= 0.0 and hi <= 1.0:
            lo, hi = 0.0, 1.0
        elif lo >= 0.0 and hi <= 255.0:
            lo, hi = 0.0, 255.0
        elif hi <= lo:
            lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.uint8)
    out = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(out * 255.0).astype(np.uint8)


def normalize_to_uint8(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.uint8)
    out = (gray - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


def raw_to_uint8(gray: np.ndarray) -> np.ndarray:
    """Convert to 8-bit for display without percentile contrast stretching."""
    gray = np.asarray(gray, dtype=np.float32)
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    cleaned = np.nan_to_num(
        gray,
        nan=float(np.nanmedian(finite)),
        posinf=float(np.nanmax(finite)),
        neginf=float(np.nanmin(finite)),
    )
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if min_value >= 0.0 and max_value <= 255.0:
        return np.clip(cleaned, 0, 255).astype(np.uint8)
    if min_value >= 0.0 and max_value <= 65535.0:
        return np.clip(cleaned / 257.0, 0, 255).astype(np.uint8)
    if max_value <= min_value:
        return np.zeros_like(gray, dtype=np.uint8)
    out = (cleaned - min_value) / (max_value - min_value)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
