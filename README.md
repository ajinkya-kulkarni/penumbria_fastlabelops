# penumbria_fastlabelops

Three focused compiled CPU operations for the current [Penumbria](https://github.com/postnubilaphoebus/Penumbria) 3D instance-segmentation pipeline. This is intentionally **not** a general 3D image-processing package.

The package targets three concrete Penumbria hot paths:

1. `label_bboxes_3d` replaces repeated whole-volume `argwhere(labels == id)` scans used to discover each training instance's extent.
2. `watershed_3d` specializes the exact simple 3D watershed mode currently used by Penumbria, avoiding generic scikit-image wrapper allocations while preserving its labels exactly on the equivalence suite.
3. `filter_instances_3d` fuses Penumbria's post-watershed size check, maximum-confidence check, and compact relabeling.

NumPy is the only runtime dependency. The hot paths are implemented in a small C++17 extension. SciPy and scikit-image are used only by benchmarks/tests as reference implementations.

## Install

```bash
pip install .
```

## API

### `label_bboxes_3d`

```python
from penumbria_fastlabelops import label_bboxes_3d

ids, boxes = label_bboxes_3d(labels)
```

`labels` is a 3D `int32`, `uint32`, `int64`, or `uint64` instance volume with `0` as background. The result contains observed foreground IDs in ascending order and one bounding box per ID:

```text
(z0, y0, x0, z1, y1, x1)
```

Upper bounds are exclusive. Penumbria can then build each local object mask from its small crop instead of scanning the full volume once per label.

The implementation is intentionally optimized for Penumbria-style compact instance IDs: internal bookkeeping scales with `max(label)`. Arbitrary highly sparse label spaces are outside the supported use case, and IDs larger than the voxel count are rejected defensively.

### `watershed_3d`

```python
from penumbria_fastlabelops import watershed_3d

labels = watershed_3d(
    prediction,
    markers,
    background_threshold=0.2,
)
```

This is deliberately specialized to Penumbria's current 3D call:

```python
background_image = (prediction > background_threshold).astype(int)
labels = skimage.segmentation.watershed(
    -prediction,
    markers,
    mask=background_image,
)
```

It implements only the semantics Penumbria currently uses: supplied markers, 6-neighbor 3D connectivity, `compactness=0`, and `watershed_line=False`. It preserves scikit-image's priority-queue flood, marker insertion order, neighbor order, flood-level propagation, and age tie-break used for plateaus.

The specialized path works directly from Penumbria's float32 prediction and does **not** materialize the negated prediction, explicit integer background mask, scikit-image float64 image conversion, padded image/marker/mask arrays, or final crop copy. The output label volume and priority queue are still required.

### `filter_instances_3d`

```python
from penumbria_fastlabelops import filter_instances_3d

filtered = filter_instances_3d(
    watershed_labels,
    prediction,
    minimum_cell_size=9,
    cell_confidence_minimum=0.51,
)
```

The behavior intentionally matches Penumbria's current post-watershed rules:

- keep an instance only when `voxel_count > minimum_cell_size`
- keep it only when its maximum prediction value is `> cell_confidence_minimum`
- reject an instance containing a NaN prediction value, matching Penumbria's NumPy top-1 comparison
- compact survivors to `1..N` in ascending original-label order
- preserve `0` as background

## Exact-equivalence validation

The watershed reference tests compare with `np.testing.assert_array_equal`, not a tolerance. They cover random continuous prediction fields, perfectly flat plateaus, quantized ties, multi-voxel markers, mask holes, markers outside the mask, non-contiguous inputs, and all supported marker integer dtypes.

The two label-operation benchmarks also assert exact output equality against their Penumbria-style NumPy/SciPy baselines before reporting timings.

This establishes exact equivalence for the three helper operations on the test suite. A full end-to-end Penumbria heatmap comparison on a real labeled volume is a separate validation step and is not claimed here.

## Penumbria integration

See [`docs/PENUMBRIA_INTEGRATION.md`](docs/PENUMBRIA_INTEGRATION.md) for the three minimal call-site changes. The guide is pinned to Penumbria commit `a4f869354a9b198f2fdf6ff2122ef1d31541b8aa`, preserves the existing local EDT construction, and keeps the original 2D postprocessing path as a fallback.

## Speed

The benchmarks reproduce the relevant Penumbria code patterns and verify exact output equivalence before reporting timings.

Representative results on synthetic **64 × 256 × 256 = 4.19M voxel** workloads:

| Penumbria operation | Baseline | Compiled | Speedup |
| --- | ---: | ---: | ---: |
| bbox discovery (`argwhere` once per label) | ~6.1 s | ~0.008 s | **~800×** |
| 3D watershed (exact Penumbria + scikit-image call) | 1.9166 s | 1.6607 s | **1.15×** |
| post-watershed filter (`find_objects` + per-instance NumPy) | ~0.028 s | ~0.009 s | **~3×** |

These are **operation-level** measurements, not end-to-end Penumbria training or inference speedups. Hardware and morphology affect absolute timings. The watershed CPU speedup is intentionally reported as modest; its additional value is avoiding several full-volume temporary arrays from the generic call path.

Reproduce locally:

```bash
python benchmarks/benchmark_penumbria.py
python benchmarks/benchmark_watershed.py --shape 64,256,256 --repeats 2
```

## Scope

Deliberately included:

- 3D bounding-box discovery needed by Penumbria data preparation
- Penumbria's exact current simple 3D watershed mode
- Penumbria-equivalent post-watershed filtering and compact relabeling

Deliberately excluded:

- generic/configurable watershed APIs
- GPU watershed
- metrics / matching
- generic regionprops
- arbitrary label reductions
- 2D APIs
- chunked/out-of-core processing

If Penumbria does not need an operation, it does not belong here.

## Development

```bash
python setup.py build_ext --inplace
PYTHONPATH=src pytest -q
PYTHONPATH=src python benchmarks/benchmark_penumbria.py
PYTHONPATH=src python benchmarks/benchmark_watershed.py
```

CI checks the minimum declared Python version, Linux/macOS/Windows builds, exact scikit-image watershed equivalence, Ruff, source/wheel packaging, and the watershed benchmark.

This repository is an independent companion implementation and does not vendor or copy Penumbria source code.
