# Penumbria integration

This repository is a companion implementation. It does not vendor or modify Penumbria source code.
The notes below describe the two minimal call-site changes audited against Penumbria commit
`a4f869354a9b198f2fdf6ff2122ef1d31541b8aa` (the `main` head on 2026-09-05).
Re-check the call sites if Penumbria moves beyond that revision.

Penumbria is currently published for review only under an all-rights-reserved license. Confirm that you
have permission to modify/use Penumbria before applying these changes.

## Install the helper

Until this package is published, install it from GitHub in the same environment as Penumbria:

```bash
pip install "git+https://github.com/ajinkya-kulkarni/penumbria_fastlabelops.git"
```

Then add the imports needed by each file as described below.

## 1. `1_prepare_training_data.py`: discover all 3D bounding boxes once

Add:

```python
from penumbria_fastlabelops import label_bboxes_3d
```

The current preprocessing loop searches the complete 3D label volume separately for every possible
label ID. Replace that label-discovery loop with the following structure after `label_heat` is
allocated:

```python
label_ids, label_boxes = label_bboxes_3d(label_img)

for i, box in tqdm(zip(label_ids, label_boxes), total=len(label_ids)):
    i = int(i)
    if i < minimum_foreground_label:
        continue

    z0, y0, x0, z1, y1, x1 = map(int, box)

    # Same local binary object used by the original EDT path, but created from
    # the small bounding-box crop rather than a whole-volume np.argwhere scan.
    local_object = label_img[z0:z1, y0:y1, x0:x1] == i
    bounding_box = np.pad(local_object.astype(np.float64), 1, mode="constant")

    heat_values, points_to_return = transform_shape_to_edt(bounding_box)
    points_to_return += np.array([z0 - 1, y0 - 1, x0 - 1])

    if heat_values is not None and points_to_return.size > 3 and heat_values.size > 0:
        label_heat[
            points_to_return[:, 0],
            points_to_return[:, 1],
            points_to_return[:, 2],
        ] = heat_values
```

Why this is equivalent to the current construction:

- returned boxes use exclusive upper bounds, so the crop is exactly the object's old min/max extent;
- `np.pad(..., 1)` recreates the existing one-voxel border around the local object;
- adding `[z0 - 1, y0 - 1, x0 - 1]` maps local EDT coordinates back to the same global coordinates;
- IDs are returned in ascending order, matching the effective order of the current integer-range loop;
- absent/gappy IDs are skipped automatically instead of triggering another full-volume scan.

The expensive part changes from approximately one full-volume scan per possible label to one compiled
scan for all observed labels, followed only by small per-object crops required by Penumbria's EDT.

## 2. `postprocess.py`: use the fused 3D instance filter

Add:

```python
from penumbria_fastlabelops import filter_instances_3d
```

Immediately after Penumbria produces the watershed label image (`wts`), use the compiled path for 3D:

```python
if prediction.ndim == 3:
    return filter_instances_3d(
        wts,
        prediction,
        minimum_cell_size=minimum_cell_size,
        cell_confidence_minimum=cell_confidence_minimum,
    )
```

Keep Penumbria's existing object-wise filtering block below it as the 2D fallback. This companion
package intentionally has no 2D API.

The 3D compiled function preserves the current rules exactly:

- `voxel_count > minimum_cell_size` (strictly greater);
- the maximum prediction value must be `> cell_confidence_minimum` (strictly greater);
- an instance containing a NaN score is rejected, matching Penumbria's NumPy top-1 comparison;
- survivors receive compact IDs `1..N` in ascending original-label order;
- background stays `0`.

## Expected operation-level speedup

The repository benchmark uses the corresponding Penumbria-style baselines and verifies output
equivalence before timing. On the documented synthetic 64 x 256 x 256 (4.19M voxel), 500-instance
workload, representative conservative results are:

| Call site | Penumbria-style baseline | `penumbria_fastlabelops` | Speedup |
| --- | ---: | ---: | ---: |
| preprocessing bbox discovery | ~6.1 s | ~0.008 s | ~800x |
| 3D post-watershed filtering | ~0.028 s | ~0.009 s | ~3x |

These are speedups for these two operations only, not end-to-end Penumbria training or inference.
Run `python benchmarks/benchmark_penumbria.py` on the target machine for local numbers.
