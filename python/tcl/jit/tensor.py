import tvm


class Tensor:

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self._data: tvm.runtime.NDArray = None

    @property
    def data(self) -> tvm.runtime.NDArray:
        return self._data

    @data.setter
    def data(self, data: "torch.Tensor"):  # type: ignore
        def _from_dlpack(tensor):
            from tvm.runtime import Device
            from tvm.runtime import ndarray

            try:
                return ndarray.from_dlpack(tensor)
            except RuntimeError:
                pass
            device_type = tensor.device.type
            device_id = tensor.device.index or 0
            return ndarray.array(
                tensor.numpy(),
                device=tvm.device(device_type, device_id),
            )

        data = _from_dlpack(data)
        assert data.shape == tuple(self.shape), "Data shape does not match tensor shape"
        assert data.dtype == self.dtype, "Data dtype does not match tensor dtype"
        self._data = data

    def __str__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"
