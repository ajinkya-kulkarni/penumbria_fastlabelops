# penumbria_fastlabelops

Two focused compiled CPU operations for the current [Penumbria](https://github.com/postnubilaphoebus/Penumbria) 3D instance-segmentation pipeline. This is intentionally **not** a general 3D image-processing package.

The package targets two concrete Penumbria bottlenecks:

1. `label_bboxes_3d` replaces repeated whole-volume `argwhere(labels == id)` scans used to discover each training instance's extent.
2. `filter_instances_3d` fuses Penumbria's post-watershed size check, maximum-confidence check, and compact relabeling.

NumPy is the only runtime dependency. The hot paths are implemented in a small C++17 extension.

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

The implementation is intentionally optimized for Penumbria-style compact instance IDs. Extremely sparse external IDs larger than the number of voxels are rejected rather than causing pathological allocations.

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

## Penumbria integration

See [`docs/PENUMBRIA_INTEGRATION.md`](docs/PENUMBRIA_INTEGRATION.md) for the two minimal call-site changes. The guide is pinned to the Penumbria revision it was audited against, preserves Penumbria's existing local EDT construction, and keeps the original 2D postprocessing path as a fallback.

## Speed

The benchmark reproduces the relevant current Penumbria code patterns and verifies exact output equivalence before reporting timings.

Representative development-machine result for a synthetic **64 × 256 × 256 = 4.19M voxel** volume with 500 requested instances:

| Penumbria operation | Baseline | Compiled | Speedup |
| --- | ---: | ---: | ---: |
| bbox discovery (`argwhere` once per label) | ~6.1 s | ~0.008 s | **~800×** |
| post-watershed filter (`find_objects` + per-instance NumPy) | ~0.028 s | ~0.009 s | **~3×** |

These are **operation-level** speedups, not end-to-end Penumbria training or inference speedups. The bbox number is large because the existing preprocessing pattern rescans the entire 3D volume for every label. Hardware and label morphology will change the absolute numbers.

Reproduce locally:

```bash
python benchmarks/benchmark_penumbria.py
```

The benchmark requires SciPy only for the reference Penumbria-style `find_objects` baseline.

## Scope

Deliberately included:

- 3D bounding-box discovery needed by Penumbria data preparation
- Penumbria-equivalent post-watershed filtering and compact relabeling

Deliberately excluded:

- watershed
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
```

CI additionally checks the minimum declared Python version, Linux/macOS/Windows builds, Ruff, and source/wheel packaging.

This repository is an independent companion implementation and does not vendor or copy Penumbria source code.
