"""Benchmark Penumbria's exact scikit-image watershed call against watershed_3d."""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.ndimage import gaussian_filter, label
from skimage.segmentation import watershed as skimage_watershed

from penumbria_fastlabelops import watershed_3d


def make_case(shape=(32, 128, 128), seed=13):
    rng = np.random.default_rng(seed)
    prediction = rng.random(shape, dtype=np.float32)
    prediction = gaussian_filter(prediction, sigma=1.5).astype(np.float32, copy=False)
    prediction -= prediction.min()
    prediction /= prediction.max() + np.float32(1e-8)

    markers, _ = label(prediction > np.float32(0.72))
    markers = markers.astype(np.int32, copy=False)
    background_threshold = 0.18
    return prediction, markers, background_threshold


def baseline(prediction, markers, background_threshold):
    threshold = np.float32(background_threshold)
    background_image = (prediction > threshold).astype(int)
    return skimage_watershed(
        -prediction,
        markers,
        mask=background_image,
    )


def timed(fn, *args, repeats=3):
    values = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        values.append(time.perf_counter() - t0)
    return min(values), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="32,128,128")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    shape = tuple(int(v) for v in args.shape.split(","))

    prediction, markers, background_threshold = make_case(shape)
    print(
        f"volume: {shape} = {prediction.size / 1e6:.2f}M voxels; "
        f"markers: {int(markers.max())}"
    )

    watershed_3d(prediction, markers, background_threshold)

    old_t, old_result = timed(
        baseline,
        prediction,
        markers,
        background_threshold,
        repeats=args.repeats,
    )
    new_t, new_result = timed(
        watershed_3d,
        prediction,
        markers,
        background_threshold,
        repeats=args.repeats,
    )
    np.testing.assert_array_equal(old_result, new_result)

    print(
        "watershed_3d: "
        f"Penumbria/skimage {old_t:.4f}s | fast {new_t:.4f}s | "
        f"speedup {old_t / new_t:.2f}x"
    )


if __name__ == "__main__":
    main()
