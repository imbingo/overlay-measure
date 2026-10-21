from __future__ import annotations
import cv2
import numpy as np
from .models import Registration, transform_points


def rigid_from_points(a_px, b_px, calibration_a, calibration_b, *, max_residual_um=0.2):
    a = calibration_a.physical(a_px)
    b = calibration_b.physical(b_px)
    if (
        a.shape != b.shape
        or len(a) < 3
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("需要至少三对有限且一一对应的独立基准点")
    ac = a - a.mean(0)
    bc = b - b.mean(0)
    if (
        min(
            np.linalg.svd(ac, compute_uv=False)[-1],
            np.linalg.svd(bc, compute_uv=False)[-1],
        )
        < 1e-6
    ):
        raise ValueError("基准点重合或共线，无法可靠检验刚性配准")
    u, _, vt = np.linalg.svd(bc.T @ ac)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = a.mean(0) - rotation @ b.mean(0)
    residual = float(
        np.sqrt(np.mean(np.sum((transform_points(b, matrix) - a) ** 2, axis=1)))
    )
    # A tiny constellation can hide a gross angular error; residual relative to span matters.
    span = float(np.sqrt(np.mean(np.sum(ac**2, axis=1))))
    reliable = residual <= max_residual_um and residual <= max(span * 0.02, 1e-8)
    return Registration(
        matrix,
        "manual_rigid",
        f"{len(a)} 对独立对应点",
        residual,
        reliable=reliable,
        warning="" if reliable else "基准残差过大，禁止定量比较",
        points_a_px=np.asarray(a_px).tolist(),
        points_b_px=np.asarray(b_px).tolist(),
    )


def translation_from_region(image_a, image_b, calibration_a, calibration_b, region):
    """Translation on a user-declared independent feature region (registration only)."""
    a = np.asarray(image_a, np.float32)
    b = np.asarray(image_b, np.float32)
    x, y, w, h = map(int, region)
    if w < 16 or h < 16 or x < 0 or y < 0 or x + w > a.shape[1] or y + h > a.shape[0]:
        raise ValueError("配准基准区须位于 A 内部且至少 16×16 px")
    # Resampling is confined to registration, never used for edge detection.
    scale = np.diag(
        [calibration_b.x / calibration_a.x, calibration_b.y / calibration_a.y, 1.0]
    )
    bb = cv2.warpAffine(
        b, scale[:2], (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0
    )
    support = cv2.warpAffine(
        np.ones(b.shape, np.uint8),
        scale[:2],
        (a.shape[1], a.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    overlap = float(support[y : y + h, x : x + w].mean())
    if overlap < 0.95:
        raise ValueError(
            "独立基准区在 B 中的共同视场不足 95%，请重新选择对应点或基准区"
        )
    aa = a[y : y + h, x : x + w].copy()
    bb = bb[y : y + h, x : x + w].copy()
    if min(float(aa.std()), float(bb.std())) < 1e-6:
        raise ValueError("配准区域缺少有效灰度特征")
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(aa.copy(), bb.copy(), win)
    initial_shift = shift
    # Refine phase's centroid estimate with a translation-only intensity fit.
    # Input images here are registration scratch buffers, not analysis images.
    warp = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], np.float32)
    score = None
    try:
        template = (aa - aa.mean()) / max(float(aa.std()), 1e-6)
        moving = (bb - bb.mean()) / max(float(bb.std()), 1e-6)
        score, warp = cv2.findTransformECC(
            template,
            moving,
            warp,
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 150, 1e-7),
            None,
            5,
        )
        shift = (float(warp[0, 2]), float(warp[1, 2]))
    except cv2.error:
        score = None
    matrix = np.eye(3)
    matrix[:2, 2] = [-shift[0] * calibration_a.x, -shift[1] * calibration_a.y]
    # Phase response is a confidence heuristic, not a metrological residual.
    reliable = bool(
        np.isfinite([*shift, response]).all()
        and response >= 0.3
        and score is not None
        and score >= 0.6
        and abs(shift[0]) < w * 0.25
        and abs(shift[1]) < h * 0.25
    )
    return Registration(
        matrix,
        "feature_translation",
        f"A 独立基准区 {x},{y},{w},{h}",
        response=float(response),
        reliable=reliable,
        warning=(
            "相位响应仅为相关可靠性指标，不等于几何残差或测量不确定度"
            if reliable
            else "相关响应低、平移细化失败或位移过大，请使用手工独立对应点"
        ),
        details=dict(
            phase_response=float(response),
            ecc_score=None if score is None else float(score),
            initial_shift_px=list(initial_shift),
            refined_shift_px=list(shift),
            algorithm="phase correlation + translation-only ECC",
            region_overlap=overlap,
        ),
    )
