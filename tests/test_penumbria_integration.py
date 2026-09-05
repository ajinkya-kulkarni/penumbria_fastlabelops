import numpy as np

from penumbria_fastlabelops import label_bboxes_3d


def _old_local_box(labels: np.ndarray, label_id: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.argwhere(labels == label_id)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    shape = tuple((maxs - mins + 3).tolist())
    box = np.zeros(shape, dtype=np.float64)
    shifted = points - (mins - 1)
    box[shifted[:, 0], shifted[:, 1], shifted[:, 2]] = 1.0
    return box, mins


def test_bbox_crop_recreates_penumbria_local_object() -> None:
    labels = np.zeros((8, 9, 10), dtype=np.int32)
    labels[0:2, 2:5, 3:7] = 2  # touches one volume boundary
    labels[4:7, 5:8, 1:3] = 19  # deliberately gappy label IDs
    labels[6, 1, 8] = 101

    ids, boxes = label_bboxes_3d(labels)

    assert ids.tolist() == [2, 19, 101]

    for label_id, box in zip(ids.tolist(), boxes.tolist(), strict=True):
        z0, y0, x0, z1, y1, x1 = box
        new_local_box = np.pad(
            (labels[z0:z1, y0:y1, x0:x1] == label_id).astype(np.float64),
            1,
            mode="constant",
        )
        old_local_box, old_mins = _old_local_box(labels, label_id)

        np.testing.assert_array_equal(new_local_box, old_local_box)
        np.testing.assert_array_equal(np.array([z0, y0, x0]), old_mins)
