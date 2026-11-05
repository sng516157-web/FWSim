"""Minimal PyTorch .pt loader without torch (M6.75 dataset task).

torch.save writes a zip: data.pkl (pickle describing tensors via
torch._utils._rebuild_tensor_v2) + data/<n> raw storages. This shim fakes the
torch symbols and reads storages as numpy — enough for plain float32/64
tensors and lists thereof (the Beatus-lab dataset layout). Deliberately
narrow: not a general torch replacement.
"""

from __future__ import annotations

import pickle
import zipfile

import numpy as np

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64,
           "LongStorage": np.int64, "IntStorage": np.int32}


class _Storage:
    def __init__(self, dtype_name):
        self.dtype_name = dtype_name
        self.data = b""

    def to_numpy(self):
        return np.frombuffer(self.data, dtype=_DTYPES[self.dtype_name])


def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                       requires_grad=False, backward_hooks=None, **_):
    arr = storage.to_numpy()
    n = int(np.prod(size))
    return arr[storage_offset:storage_offset + n].reshape(size)


def load_pt(path) -> list[np.ndarray] | np.ndarray:
    """Load a torch .pt file containing tensors or a list of tensors."""
    z = zipfile.ZipFile(path)
    names = z.namelist()
    pkl_name = next(n for n in names if n.endswith("data.pkl"))
    # storages are keyed by 'data/<key>' relative to the pkl's directory
    prefix = pkl_name.rsplit("/", 1)[0] + "/data/"

    storages: dict[str, _Storage] = {}

    def load_storage(dtype_name, key, _location):
        if key not in storages:
            st = _Storage(dtype_name)
            st.data = z.read(prefix + str(key))
            storages[key] = st
        return storages[key]

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            # torch stores storages as ('storage', StorageType, key, location, size)
            _tag, storage_type, key, _location, _size = pid
            return load_storage(storage_type.__name__, key, None)

        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if module == "torch._utils" and name == "_rebuild_tensor":
                def _rebuild_tensor(storage, storage_offset, size, stride):
                    return _rebuild_tensor_v2(storage, storage_offset, size, stride)
                return _rebuild_tensor
            if module == "torch" and name in _DTYPES:
                return type(name, (_Storage,), {"__init__": lambda self: _Storage.__init__(self, name)})
            if module == "torch._utils" and name == "_load_from_bytes":
                return lambda b: pickle.load(io.BytesIO(b))
            if module.startswith("torch"):
                raise pickle.UnpicklingError(f"unsupported torch symbol {module}.{name}")
            return super().find_class(module, name)

    import io

    return _Unpickler(io.BytesIO(z.read(pkl_name))).load()
