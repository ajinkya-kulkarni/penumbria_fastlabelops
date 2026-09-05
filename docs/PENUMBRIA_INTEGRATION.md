# Penumbria integration

This repository is a companion implementation. It does not vendor or modify Penumbria source code.
The notes below describe three focused call-site changes audited against Penumbria commit
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

## 2. `postprocess.py`: replace the generic 3D watershed wrapper

Add:

```python
from penumbria_fastlabelops import watershed_3d
```

For the current 3D path, replace:

```python
background_image = (prediction > background_threshold).astype(int)
wts = watershed(-prediction, labeled_array, mask=background_image)
```

with:

```python
if prediction.ndim == 3:
    wts = watershed_3d(
        prediction,
        labeled_array,
        background_threshold=background_threshold,
    )
else:
    background_image = (prediction > background_threshold).astype(int)
    wts = watershed(-prediction, labeled_array, mask=background_image)
```

`watershed_3d` is intentionally specialized to exactly the scikit-image mode Penumbria currently
uses: supplied markers, default connectivity (6-neighbor in 3D), `compactness=0`, and
`watershed_line=False`. It preserves marker values and uses the same priority-queue flood and age
tie-break behavior.

The specialized implementation does not materialize the negated prediction, explicit integer
background mask, float64 image conversion, padded image/marker/mask arrays, or final crop copy required
by the generic scikit-image wrapper.

## 3. `postprocess.py`: use the fused 3D instance filter

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

## Validation and speed

All equivalence tests use exact array equality. The watershed suite includes random continuous fields,
flat plateaus, quantized ties, multi-voxel markers, mask holes, masked-out markers, non-contiguous
inputs, and all supported marker integer dtypes.

The benchmarks use the corresponding Penumbria-style baselines and verify output equivalence before
timing. Representative 64 x 256 x 256 (4.19M voxel) results are:

| Call site | Penumbria-style baseline | `penumbria_fastlabelops` | Speedup |
| --- | ---: | ---: | ---: |
| preprocessing bbox discovery | ~6.1 s | ~0.008 s | ~800x |
| 3D watershed | 1.9166 s | 1.6607 s | 1.15x |
| 3D post-watershed filtering | ~0.028 s | ~0.009 s | ~3x |

The watershed CPU speedup is modest; its additional benefit is avoiding several full-volume temporary
arrays in Penumbria's generic scikit-image call path.

Run `python benchmarks/benchmark_penumbria.py` for the two label operations and
`python benchmarks/benchmark_watershed.py --shape 64,256,256 --repeats 2` for the watershed comparison.
These are operation-level measurements, not end-to-end Penumbria training or inference claims.
