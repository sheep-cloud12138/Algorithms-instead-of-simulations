"""Read a torch.save() .pt file with numpy only (zip + pickle + raw storages)."""
import zipfile, pickle, collections, numpy as np

_DT = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'HalfStorage': np.float16,
       'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
       'ByteStorage': np.uint8, 'BoolStorage': np.bool_, 'CharStorage': np.int8}


class _Storage:
    def __init__(self, dtype):
        self.dtype = dtype
        self.data = None
        self.numel = 0


def _rebuild(storage, offset, size, stride, *args):
    flat = storage.data
    if flat is None:
        raise RuntimeError('storage not loaded')
    n = int(np.prod(size)) if len(size) else 1
    if len(size) == 0:
        return flat[offset]
    idx = np.zeros(size, dtype=np.int64)
    # build indices from strides
    grids = np.indices(size, sparse=False)
    for ax, (s, st) in enumerate(zip(size, stride)):
        idx = idx + grids[ax] * st
    return flat[offset + idx]


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zipf, prefix):
        super().__init__(f)
        self.zipf = zipf
        self.prefix = prefix
        self.storages = {}

    def find_class(self, module, name):
        if module == 'torch' and name.endswith('Storage'):
            dtype = _DT[name]
            return type(name, (_Storage,), {'storage_dtype': dtype,
                                            '__init__': lambda self, d=dtype: _Storage.__init__(self, d)})
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return _rebuild
        if module == 'torch' and name == 'Size':
            return tuple
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        return super().find_class(module, name)

    def persistent_load(self, pid):
        kind, dtype_name, key, location, numel = pid[:5]
        assert kind == 'storage', kind
        if isinstance(dtype_name, _Storage):
            dtype = dtype_name.dtype
        elif isinstance(dtype_name, type) and hasattr(dtype_name, 'storage_dtype'):
            dtype = dtype_name.storage_dtype
        else:
            dtype = _DT[dtype_name]
        st = _Storage(dtype)
        st.numel = numel
        st.data = np.frombuffer(self.zipf.read(f'{self.prefix}data/{key}'), dtype=st.dtype, count=numel)
        self.storages[key] = st
        return st


def load(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith('data.pkl')][0]
    prefix = pkl[:-len('data.pkl')]
    with zf.open(pkl) as f:
        obj = _Unpickler(f, zf, prefix).load()
    return obj
