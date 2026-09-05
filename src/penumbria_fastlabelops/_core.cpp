#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {

bool labels_type_ok(int typenum) {
    return typenum == NPY_INT32 || typenum == NPY_UINT32 || typenum == NPY_INT64 ||
           typenum == NPY_UINT64;
}

bool scores_type_ok(int typenum) {
    return typenum == NPY_FLOAT32 || typenum == NPY_FLOAT64;
}

bool validate_labels(PyArrayObject* arr) {
    if (PyArray_NDIM(arr) != 3) {
        PyErr_SetString(PyExc_ValueError, "labels must be 3D");
        return false;
    }
    if (!labels_type_ok(PyArray_TYPE(arr))) {
        PyErr_SetString(PyExc_TypeError, "unsupported labels dtype");
        return false;
    }
    if (!PyArray_ISCARRAY_RO(arr)) {
        PyErr_SetString(PyExc_ValueError, "labels must be C-contiguous");
        return false;
    }
    return true;
}

template <typename T>
void grow_to(std::vector<T>& values, size_t index, const T& fill) {
    if (index < values.size()) {
        return;
    }
    size_t next = values.empty() ? 16 : values.size();
    while (next <= index) {
        if (next > std::numeric_limits<size_t>::max() / 2) {
            next = index + 1;
            break;
        }
        next *= 2;
    }
    values.resize(next, fill);
}

template <typename LabelT>
PyObject* bboxes_impl(PyArrayObject* labels) {
    const auto* data = static_cast<const LabelT*>(PyArray_DATA(labels));
    const npy_intp* dims = PyArray_DIMS(labels);
    const uint64_t nvox = static_cast<uint64_t>(PyArray_SIZE(labels));
    const int64_t inf = std::numeric_limits<int64_t>::max();

    std::vector<int64_t> min_z(16, inf), min_y(16, inf), min_x(16, inf);
    std::vector<int64_t> max_z(16, -1), max_y(16, -1), max_x(16, -1);
    std::vector<uint8_t> seen(16, 0);
    size_t max_id = 0;
    bool invalid_label = false;
    bool allocation_failed = false;

    const int64_t D = static_cast<int64_t>(dims[0]);
    const int64_t H = static_cast<int64_t>(dims[1]);
    const int64_t W = static_cast<int64_t>(dims[2]);

    Py_BEGIN_ALLOW_THREADS
    try {
        for (int64_t z = 0; z < D; ++z) {
            for (int64_t y = 0; y < H; ++y) {
                const npy_intp base = (z * H + y) * W;
                for (int64_t x = 0; x < W; ++x) {
                    const LabelT raw = data[base + x];
                    if constexpr (std::numeric_limits<LabelT>::is_signed) {
                        if (raw < 0) {
                            invalid_label = true;
                            continue;
                        }
                    }
                    const uint64_t id64 = static_cast<uint64_t>(raw);
                    if (id64 == 0) {
                        continue;
                    }
                    if (id64 > nvox) {
                        invalid_label = true;
                        continue;
                    }
                    const size_t id = static_cast<size_t>(id64);
                    grow_to(min_z, id, inf);
                    grow_to(min_y, id, inf);
                    grow_to(min_x, id, inf);
                    grow_to(max_z, id, int64_t{-1});
                    grow_to(max_y, id, int64_t{-1});
                    grow_to(max_x, id, int64_t{-1});
                    grow_to(seen, id, uint8_t{0});
                    max_id = std::max(max_id, id);
                    seen[id] = 1;
                    min_z[id] = std::min(min_z[id], z);
                    min_y[id] = std::min(min_y[id], y);
                    min_x[id] = std::min(min_x[id], x);
                    max_z[id] = std::max(max_z[id], z + 1);
                    max_y[id] = std::max(max_y[id], y + 1);
                    max_x[id] = std::max(max_x[id], x + 1);
                }
            }
        }
    } catch (const std::bad_alloc&) {
        allocation_failed = true;
    } catch (const std::length_error&) {
        allocation_failed = true;
    }
    Py_END_ALLOW_THREADS

    if (allocation_failed) {
        return PyErr_NoMemory();
    }
    if (invalid_label) {
        PyErr_SetString(PyExc_ValueError, "labels must be non-negative with Penumbria-style compact IDs");
        return nullptr;
    }

    npy_intp count = 0;
    for (size_t id = 1; id <= max_id; ++id) {
        count += seen[id] != 0;
    }

    npy_intp ids_dim[1] = {count};
    npy_intp bbox_dims[2] = {count, 6};
    auto* ids = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, ids_dim, PyArray_TYPE(labels)));
    auto* bboxes = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, bbox_dims, NPY_INT64));
    if (ids == nullptr || bboxes == nullptr) {
        Py_XDECREF(ids);
        Py_XDECREF(bboxes);
        return nullptr;
    }

    auto* ids_out = static_cast<LabelT*>(PyArray_DATA(ids));
    auto* bbox_out = static_cast<int64_t*>(PyArray_DATA(bboxes));
    npy_intp row = 0;
    for (size_t id = 1; id <= max_id; ++id) {
        if (!seen[id]) {
            continue;
        }
        ids_out[row] = static_cast<LabelT>(id);
        bbox_out[row * 6 + 0] = min_z[id];
        bbox_out[row * 6 + 1] = min_y[id];
        bbox_out[row * 6 + 2] = min_x[id];
        bbox_out[row * 6 + 3] = max_z[id];
        bbox_out[row * 6 + 4] = max_y[id];
        bbox_out[row * 6 + 5] = max_x[id];
        ++row;
    }

    return Py_BuildValue("NN", reinterpret_cast<PyObject*>(ids), reinterpret_cast<PyObject*>(bboxes));
}

PyObject* py_label_bboxes_3d(PyObject*, PyObject* args) {
    PyArrayObject* labels = nullptr;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &labels)) {
        return nullptr;
    }
    if (!validate_labels(labels)) {
        return nullptr;
    }
    switch (PyArray_TYPE(labels)) {
        case NPY_INT32:
            return bboxes_impl<int32_t>(labels);
        case NPY_UINT32:
            return bboxes_impl<uint32_t>(labels);
        case NPY_INT64:
            return bboxes_impl<int64_t>(labels);
        case NPY_UINT64:
            return bboxes_impl<uint64_t>(labels);
        default:
            break;
    }
    PyErr_SetString(PyExc_TypeError, "unsupported labels dtype");
    return nullptr;
}

template <typename LabelT, typename ScoreT>
PyObject* filter_impl(
    PyArrayObject* labels,
    PyArrayObject* scores,
    uint64_t minimum_cell_size,
    double confidence_minimum
) {
    const auto* label_data = static_cast<const LabelT*>(PyArray_DATA(labels));
    const auto* score_data = static_cast<const ScoreT*>(PyArray_DATA(scores));
    const npy_intp n = PyArray_SIZE(labels);

    std::vector<uint64_t> counts(16, 0);
    std::vector<double> maxima(16, -std::numeric_limits<double>::infinity());
    std::vector<uint8_t> has_nan(16, 0);
    size_t max_id = 0;
    bool invalid_label = false;
    bool allocation_failed = false;

    Py_BEGIN_ALLOW_THREADS
    try {
        for (npy_intp i = 0; i < n; ++i) {
            const LabelT raw = label_data[i];
            if constexpr (std::numeric_limits<LabelT>::is_signed) {
                if (raw < 0) {
                    invalid_label = true;
                    continue;
                }
            }
            const uint64_t id64 = static_cast<uint64_t>(raw);
            if (id64 == 0) {
                continue;
            }
            if (id64 > static_cast<uint64_t>(n)) {
                invalid_label = true;
                continue;
            }
            const size_t id = static_cast<size_t>(id64);
            grow_to(counts, id, uint64_t{0});
            grow_to(maxima, id, -std::numeric_limits<double>::infinity());
            grow_to(has_nan, id, uint8_t{0});
            max_id = std::max(max_id, id);
            ++counts[id];
            const double score = static_cast<double>(score_data[i]);
            if (std::isnan(score)) {
                has_nan[id] = 1;
            } else if (score > maxima[id]) {
                maxima[id] = score;
            }
        }
    } catch (const std::bad_alloc&) {
        allocation_failed = true;
    } catch (const std::length_error&) {
        allocation_failed = true;
    }
    Py_END_ALLOW_THREADS

    if (allocation_failed) {
        return PyErr_NoMemory();
    }
    if (invalid_label) {
        PyErr_SetString(PyExc_ValueError, "labels must be non-negative with Penumbria-style compact IDs");
        return nullptr;
    }

    std::vector<LabelT> mapping;
    try {
        mapping.assign(max_id + 1, static_cast<LabelT>(0));
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::length_error&) {
        return PyErr_NoMemory();
    }

    uint64_t next_id = 1;
    for (size_t id = 1; id <= max_id; ++id) {
        if (!has_nan[id] && counts[id] > minimum_cell_size && maxima[id] > confidence_minimum) {
            if (next_id > static_cast<uint64_t>(std::numeric_limits<LabelT>::max())) {
                PyErr_SetString(PyExc_OverflowError, "too many surviving labels for output dtype");
                return nullptr;
            }
            mapping[id] = static_cast<LabelT>(next_id++);
        }
    }

    const npy_intp* dims = PyArray_DIMS(labels);
    auto* out = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(3, dims, PyArray_TYPE(labels)));
    if (out == nullptr) {
        return nullptr;
    }
    auto* out_data = static_cast<LabelT*>(PyArray_DATA(out));

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < n; ++i) {
        const uint64_t id = static_cast<uint64_t>(label_data[i]);
        out_data[i] = id == 0 ? static_cast<LabelT>(0) : mapping[static_cast<size_t>(id)];
    }
    Py_END_ALLOW_THREADS

    return reinterpret_cast<PyObject*>(out);
}

PyObject* py_filter_instances_3d(PyObject*, PyObject* args) {
    PyArrayObject* labels = nullptr;
    PyArrayObject* scores = nullptr;
    unsigned long long minimum_cell_size = 0;
    double confidence_minimum = 0.0;
    if (!PyArg_ParseTuple(
            args,
            "O!O!Kd",
            &PyArray_Type,
            &labels,
            &PyArray_Type,
            &scores,
            &minimum_cell_size,
            &confidence_minimum)) {
        return nullptr;
    }
    if (!validate_labels(labels)) {
        return nullptr;
    }
    if (PyArray_NDIM(scores) != 3 || !scores_type_ok(PyArray_TYPE(scores)) ||
        !PyArray_ISCARRAY_RO(scores)) {
        PyErr_SetString(PyExc_ValueError, "scores must be contiguous float32 or float64 3D array");
        return nullptr;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (PyArray_DIMS(labels)[axis] != PyArray_DIMS(scores)[axis]) {
            PyErr_SetString(PyExc_ValueError, "labels and scores shapes must match");
            return nullptr;
        }
    }

#define DISPATCH_SCORE(LABEL_T) \
    if (PyArray_TYPE(scores) == NPY_FLOAT32) \
        return filter_impl<LABEL_T, float>(labels, scores, minimum_cell_size, confidence_minimum); \
    return filter_impl<LABEL_T, double>(labels, scores, minimum_cell_size, confidence_minimum)

    switch (PyArray_TYPE(labels)) {
        case NPY_INT32:
            DISPATCH_SCORE(int32_t);
        case NPY_UINT32:
            DISPATCH_SCORE(uint32_t);
        case NPY_INT64:
            DISPATCH_SCORE(int64_t);
        case NPY_UINT64:
            DISPATCH_SCORE(uint64_t);
        default:
            break;
    }
#undef DISPATCH_SCORE

    PyErr_SetString(PyExc_TypeError, "unsupported labels dtype");
    return nullptr;
}

PyMethodDef methods[] = {
    {"label_bboxes_3d", py_label_bboxes_3d, METH_VARARGS, "Compute 3D label bounding boxes."},
    {"filter_instances_3d", py_filter_instances_3d, METH_VARARGS, "Filter and compact 3D labels."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "penumbria_fastlabelops C++ core",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__core(void) {
    import_array();
    return PyModule_Create(&module);
}
