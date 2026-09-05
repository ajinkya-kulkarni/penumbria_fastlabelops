import numpy as np
import pytest

from penumbria_fastlabelops import watershed_3d

skimage_segmentation = pytest.importorskip("skimage.segmentation")
skimage_watershed = skimage_segmentation.watershed


def reference_watershed(prediction, markers, background_threshold):
    prediction = np.asarray(prediction, dtype=np.float32)
    threshold = np.float32(background_threshold)
    return skimage_watershed(
        -prediction,
        markers,
        mask=prediction > threshold,
    )


@pytest.mark.parametrize("marker_dtype", [np.int32, np.uint32, np.int64, np.uint64])
def test_watershed_random_matches_skimage_bitwise(marker_dtype):
    rng = np.random.default_rng(17)
    shape = (11, 13, 15)

    for _ in range(10):
        prediction = rng.random(shape, dtype=np.float32)
        markers = np.zeros(shape, dtype=marker_dtype)
        for label_id in range(1, 9):
            coord = tuple(int(rng.integers(0, dim)) for dim in shape)
            markers[coord] = label_id * 3

        got = watershed_3d(prediction, markers, background_threshold=0.2)
        expected = reference_watershed(prediction, markers, 0.2)
        np.testing.assert_array_equal(got, expected)
        assert got.dtype == markers.dtype


def test_watershed_plateau_tie_break_matches_skimage_bitwise():
    prediction = np.ones((5, 7, 11), dtype=np.float32)
    markers = np.zeros(prediction.shape, dtype=np.int32)
    markers[2, 3, 1] = 7
    markers[2, 3, 5] = 3
    markers[2, 3, 9] = 19

    got = watershed_3d(prediction, markers, background_threshold=0.0)
    expected = reference_watershed(prediction, markers, 0.0)
    np.testing.assert_array_equal(got, expected)


def test_watershed_quantized_ties_match_skimage_bitwise():
    rng = np.random.default_rng(23)
    prediction = (rng.integers(0, 5, size=(9, 10, 11)) / 4).astype(np.float32)
    markers = np.zeros(prediction.shape, dtype=np.int32)
    markers[1, 1, 1] = 1
    markers[7, 8, 9] = 2
    markers[3, 6, 4] = 8
    markers[6, 2, 7] = 13

    got = watershed_3d(prediction, markers, background_threshold=0.25)
    expected = reference_watershed(prediction, markers, 0.25)
    np.testing.assert_array_equal(got, expected)


def test_watershed_mask_holes_and_marker_outside_mask_match_skimage():
    prediction = np.ones((6, 8, 10), dtype=np.float32)
    prediction[:, 4, :] = 0.0
    prediction[1:3, 1:3, 1:3] = 0.1

    markers = np.zeros(prediction.shape, dtype=np.int32)
    markers[2, 2, 2] = 99  # outside mask, must disappear
    markers[2, 1, 6] = 4
    markers[4, 6, 7] = 11

    got = watershed_3d(prediction, markers, background_threshold=0.5)
    expected = reference_watershed(prediction, markers, 0.5)
    np.testing.assert_array_equal(got, expected)


def test_watershed_non_contiguous_inputs_match_skimage():
    rng = np.random.default_rng(31)
    prediction = rng.random((8, 9, 10), dtype=np.float32)[:, :, ::-1]
    markers = np.zeros((8, 9, 10), dtype=np.int32)
    markers[1, 2, 3] = 1
    markers[6, 7, 8] = 5
    markers = markers[:, :, ::-1]

    got = watershed_3d(prediction, markers, background_threshold=0.15)
    expected = reference_watershed(prediction, markers, 0.15)
    np.testing.assert_array_equal(got, expected)


def test_watershed_float64_input_matches_penumbria_float32_semantics():
    rng = np.random.default_rng(41)
    prediction = rng.random((7, 8, 9))
    markers = np.zeros(prediction.shape, dtype=np.int32)
    markers[1, 1, 1] = 1
    markers[5, 6, 7] = 2

    got = watershed_3d(prediction, markers, background_threshold=0.2)
    expected = reference_watershed(prediction, markers, 0.2)
    np.testing.assert_array_equal(got, expected)


def test_watershed_all_masked_returns_zero():
    prediction = np.zeros((3, 4, 5), dtype=np.float32)
    markers = np.zeros(prediction.shape, dtype=np.int32)
    markers[1, 2, 3] = 7

    got = watershed_3d(prediction, markers, background_threshold=0.2)
    np.testing.assert_array_equal(got, np.zeros_like(markers))


def test_watershed_shape_mismatch_rejected():
    prediction = np.zeros((3, 4, 5), dtype=np.float32)
    markers = np.zeros((3, 4, 6), dtype=np.int32)
    with pytest.raises(ValueError, match="does not match"):
        watershed_3d(prediction, markers)


def test_watershed_negative_markers_rejected():
    prediction = np.ones((3, 4, 5), dtype=np.float32)
    markers = np.zeros(prediction.shape, dtype=np.int32)
    markers[1, 2, 3] = -1
    with pytest.raises(ValueError, match="non-negative"):
        watershed_3d(prediction, markers)
