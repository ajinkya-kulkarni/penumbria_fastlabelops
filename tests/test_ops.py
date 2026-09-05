import numpy as np
import pytest

from penumbria_fastlabelops import filter_instances_3d, label_bboxes_3d


def reference_bboxes(labels):
    ids = np.unique(labels)
    ids = ids[ids != 0]
    boxes = []
    for label_id in ids:
        points = np.argwhere(labels == label_id)
        mins = points.min(axis=0)
        maxs = points.max(axis=0) + 1
        boxes.append(np.concatenate([mins, maxs]))
    return ids, np.asarray(boxes, dtype=np.int64).reshape(-1, 6)


def reference_filter(labels, scores, minimum_cell_size, confidence_minimum):
    out = np.zeros_like(labels)
    next_id = 1
    for label_id in range(1, int(labels.max()) + 1):
        mask = labels == label_id
        if np.count_nonzero(mask) > minimum_cell_size and np.max(
            scores[mask], initial=-np.inf
        ) > confidence_minimum:
            out[mask] = next_id
            next_id += 1
    return out


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
def test_label_bboxes_matches_reference(dtype):
    labels = np.zeros((5, 6, 7), dtype=dtype)
    labels[1:3, 2:5, 3:6] = 2
    labels[4, 0:2, 1:4] = 9
    ids, boxes = label_bboxes_3d(labels)
    ref_ids, ref_boxes = reference_bboxes(labels)
    np.testing.assert_array_equal(ids, ref_ids)
    np.testing.assert_array_equal(boxes, ref_boxes)


def test_label_bboxes_accepts_non_contiguous_input():
    labels = np.zeros((6, 7, 8), dtype=np.int32)
    labels[1:4, 2:6, 3:7] = 4
    view = labels[:, :, ::-1]
    ids, boxes = label_bboxes_3d(view)
    ref_ids, ref_boxes = reference_bboxes(view)
    np.testing.assert_array_equal(ids, ref_ids)
    np.testing.assert_array_equal(boxes, ref_boxes)


def test_label_bboxes_empty():
    labels = np.zeros((2, 3, 4), dtype=np.int32)
    ids, boxes = label_bboxes_3d(labels)
    assert ids.shape == (0,)
    assert boxes.shape == (0, 6)


def test_filter_matches_penumbria_semantics():
    labels = np.zeros((3, 4, 5), dtype=np.int32)
    labels[0, 0, 0:3] = 1
    labels[0:2, 1, 0:2] = 4
    labels[1:3, 2:4, 2:4] = 7
    labels[0:2, 3, 0:2] = 9

    scores = np.zeros(labels.shape, dtype=np.float32)
    scores[labels == 1] = 0.99
    scores[labels == 4] = 0.5
    scores[labels == 7] = 0.8
    scores[labels == 9] = 0.9

    got = filter_instances_3d(
        labels,
        scores,
        minimum_cell_size=3,
        cell_confidence_minimum=0.5,
    )
    expected = reference_filter(labels, scores, 3, 0.5)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == labels.dtype


@pytest.mark.parametrize("label_dtype", [np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize("score_dtype", [np.float32, np.float64])
def test_filter_random_matches_reference(label_dtype, score_dtype):
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 25, size=(19, 17, 13), dtype=np.int32).astype(label_dtype)
    scores = rng.random(labels.shape).astype(score_dtype)
    got = filter_instances_3d(
        labels,
        scores,
        minimum_cell_size=5,
        cell_confidence_minimum=0.85,
    )
    expected = reference_filter(labels, scores, 5, 0.85)
    np.testing.assert_array_equal(got, expected)


def test_filter_nan_scores_match_penumbria_semantics():
    labels = np.zeros((2, 3, 4), dtype=np.int32)
    labels[0, 0, :3] = 1
    labels[0, 1, :3] = 2
    labels[1, 0, :3] = 3

    scores = np.zeros(labels.shape, dtype=np.float32)
    scores[labels == 1] = [0.7, np.nan, 0.9]
    scores[labels == 2] = [0.7, 0.8, 0.9]
    scores[labels == 3] = np.nan

    got = filter_instances_3d(
        labels,
        scores,
        minimum_cell_size=2,
        cell_confidence_minimum=0.5,
    )
    expected = reference_filter(labels, scores, 2, 0.5)
    np.testing.assert_array_equal(got, expected)

    # Penumbria's top-1 comparison rejects an instance if any NaN is present,
    # because NumPy orders NaN as the largest value for argpartition/sort.
    assert np.unique(got).tolist() == [0, 1]
    assert np.all(got[labels == 2] == 1)


def test_negative_labels_rejected():
    labels = np.zeros((2, 2, 2), dtype=np.int32)
    labels[0, 0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        label_bboxes_3d(labels)


def test_filter_shape_mismatch_rejected():
    labels = np.zeros((2, 2, 2), dtype=np.int32)
    scores = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        filter_instances_3d(labels, scores)


def test_extreme_sparse_label_rejected():
    labels = np.zeros((2, 2, 2), dtype=np.int64)
    labels[0, 0, 0] = 10_000
    with pytest.raises(ValueError, match="compact IDs"):
        label_bboxes_3d(labels)
