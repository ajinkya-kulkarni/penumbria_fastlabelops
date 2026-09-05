from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import _core

_LABEL_DTYPES = (np.dtype(np.int32), np.dtype(np.uint32), np.dtype(np.int64), np.dtype(np.uint64))
_SCORE_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def _labels3d(labels: NDArray[np.integer]) -> NDArray[np.integer]:
    arr = np.asarray(labels)
    if arr.ndim != 3:
        raise ValueError(f"labels must be 3D, got {arr.ndim}D")
    if arr.dtype not in _LABEL_DTYPES:
        raise TypeError(f"labels dtype must be int32, uint32, int64, or uint64; got {arr.dtype}")
    return np.ascontiguousarray(arr)


def label_bboxes_3d(
    labels: NDArray[np.integer],
) -> tuple[NDArray[np.integer], NDArray[np.int64]]:
    """Return observed foreground label IDs and their 3D bounding boxes.

    Bounding boxes are ``(z0, y0, x0, z1, y1, x1)`` with exclusive upper bounds.
    Label 0 is background and is omitted. IDs are returned in ascending order.

    This is intentionally optimized for Penumbria-style compact labels: internal
    storage scales with the maximum label ID. Arbitrary extremely sparse IDs are
    outside the supported use case.
    """
    arr = _labels3d(labels)
    return _core.label_bboxes_3d(arr)


def watershed_3d(
    prediction: NDArray[np.floating],
    markers: NDArray[np.integer],
    background_threshold: float = 0.2,
) -> NDArray[np.integer]:
    """Run the exact simple 3D watershed mode used by Penumbria.

    This is specialized to Penumbria's current call to scikit-image:

    ``watershed(-prediction, markers, mask=prediction > background_threshold)``

    with default connectivity (6-neighbor in 3D), ``compactness=0`` and
    ``watershed_line=False``. Prediction is converted to float32 because Penumbria
    does that before watershed. Marker values are preserved; zero is background.

    The implementation avoids materializing ``-prediction``, the boolean mask,
    float64 image copies, padded image/marker/mask arrays, and the final crop copy
    used by the generic scikit-image wrapper.
    """
    prediction_arr = np.asarray(prediction, dtype=np.float32)
    if prediction_arr.ndim != 3:
        raise ValueError(f"prediction must be 3D, got {prediction_arr.ndim}D")
    prediction_arr = np.ascontiguousarray(prediction_arr)

    marker_arr = _labels3d(markers)
    if marker_arr.shape != prediction_arr.shape:
        raise ValueError(
            f"markers shape {marker_arr.shape} does not match prediction {prediction_arr.shape}"
        )

    threshold = float(np.float32(background_threshold))
    return _core.watershed_3d(prediction_arr, marker_arr, threshold)


def filter_instances_3d(
    labels: NDArray[np.integer],
    scores: NDArray[np.floating],
    minimum_cell_size: int = 9,
    cell_confidence_minimum: float = 0.5,
) -> NDArray[np.integer]:
    """Apply Penumbria's post-watershed size/confidence filter in compiled code.

    An instance is retained iff its voxel count is strictly greater than
    ``minimum_cell_size`` and its maximum score is strictly greater than
    ``cell_confidence_minimum``. An instance containing NaN is rejected, matching
    Penumbria's NumPy top-1 comparison. Retained labels are compacted to ``1..N``
    in ascending original-label order. Background remains 0.

    Labels are expected to be Penumbria-style compact instance IDs; internal
    storage scales with the maximum label ID.
    """
    label_arr = _labels3d(labels)
    score_arr = np.asarray(scores)
    if score_arr.ndim != 3:
        raise ValueError(f"scores must be 3D, got {score_arr.ndim}D")
    if score_arr.shape != label_arr.shape:
        raise ValueError(f"scores shape {score_arr.shape} does not match labels {label_arr.shape}")
    if score_arr.dtype not in _SCORE_DTYPES:
        score_arr = score_arr.astype(np.float32, copy=False)
    score_arr = np.ascontiguousarray(score_arr)
    if minimum_cell_size < 0:
        raise ValueError("minimum_cell_size must be >= 0")
    return _core.filter_instances_3d(
        label_arr,
        score_arr,
        int(minimum_cell_size),
        float(cell_confidence_minimum),
    )
