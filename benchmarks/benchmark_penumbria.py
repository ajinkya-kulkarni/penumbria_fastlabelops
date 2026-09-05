"""Benchmark the two Penumbria code patterns this package replaces."""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.ndimage import find_objects

from penumbria_fastlabelops import filter_instances_3d, label_bboxes_3d


def make_volume(shape=(64, 256, 256), instances=500, seed=7):
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, dtype=np.int32)
    scores = rng.random(shape, dtype=np.float32) * 0.15

    d, h, w = shape
    for label_id in range(1, instances + 1):
        rz = int(rng.integers(1, 4))
        ry = int(rng.integers(2, 6))
        rx = int(rng.integers(2, 6))
        z = int(rng.integers(rz, d - rz))
        y = int(rng.integers(ry, h - ry))
        x = int(rng.integers(rx, w - rx))
        labels[z - rz : z + rz + 1, y - ry : y + ry + 1, x - rx : x + rx + 1] = label_id
        scores[z - rz : z + rz + 1, y - ry : y + ry + 1, x - rx : x + rx + 1] = rng.uniform(
            0.35, 1.0
        )
    return labels, scores


def baseline_bboxes(labels):
    boxes = []
    ids = []
    for label_id in range(1, int(labels.max()) + 1):
        points = np.argwhere(labels == label_id)
        if points.size == 0:
            continue
        mins = points.min(axis=0)
        maxs = points.max(axis=0) + 1
        ids.append(label_id)
        boxes.append(np.concatenate((mins, maxs)))
    return np.asarray(ids, dtype=labels.dtype), np.asarray(boxes, dtype=np.int64).reshape(-1, 6)


def baseline_filter(labels, scores, minimum_cell_size=9, confidence_minimum=0.51):
    slices = find_objects(labels)
    out = np.zeros_like(labels)
    next_id = 1
    for label_id, slice_tuple in enumerate(slices, start=1):
        if slice_tuple is None:
            continue
        local_locs = np.array(np.where(labels[slice_tuple] == label_id))
        global_locs = np.stack(local_locs).T + np.array([sl.start for sl in slice_tuple])
        if len(global_locs) > minimum_cell_size:
            values = scores[global_locs[:, 0], global_locs[:, 1], global_locs[:, 2]]
            if values.size and float(values.max()) > confidence_minimum:
                out[global_locs[:, 0], global_locs[:, 1], global_locs[:, 2]] = next_id
                next_id += 1
    return out


def timed(fn, *args, repeats=3, **kwargs):
    values = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        values.append(time.perf_counter() - t0)
    return min(values), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="64,256,256")
    parser.add_argument("--instances", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    shape = tuple(int(v) for v in args.shape.split(","))

    labels, scores = make_volume(shape, args.instances)
    voxels = labels.size
    print(f"volume: {shape} = {voxels / 1e6:.2f}M voxels; requested instances: {args.instances}")

    warm_ids, _ = label_bboxes_3d(labels)
    filter_instances_3d(labels, scores, 9, 0.51)
    print(f"observed foreground labels: {len(warm_ids)}")

    old_t, old_result = timed(baseline_bboxes, labels, repeats=args.repeats)
    new_t, new_result = timed(label_bboxes_3d, labels, repeats=args.repeats)
    np.testing.assert_array_equal(old_result[0], new_result[0])
    np.testing.assert_array_equal(old_result[1], new_result[1])
    print(
        f"label_bboxes_3d: baseline {old_t:.4f}s | fast {new_t:.4f}s | "
        f"speedup {old_t / new_t:.1f}x"
    )

    old_t, old_result = timed(baseline_filter, labels, scores, 9, 0.51, repeats=args.repeats)
    new_t, new_result = timed(filter_instances_3d, labels, scores, 9, 0.51, repeats=args.repeats)
    np.testing.assert_array_equal(old_result, new_result)
    print(
        f"filter_instances_3d: baseline {old_t:.4f}s | fast {new_t:.4f}s | "
        f"speedup {old_t / new_t:.1f}x"
    )


if __name__ == "__main__":
    main()
