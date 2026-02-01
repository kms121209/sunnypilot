"""
The whole point of this module is to mimic compile3.py while adapting it for our buffers to prevent buffer explosion
"""
import os
import sys
import pickle

from tinygrad import Tensor, TinyJit, Context, Device
from tinygrad.device import Buffer
# from tinygrad.nn.onnx import OnnxRunner
from tinygrad.frontend.onnx import OnnxRunner

if "JIT_BATCH_SIZE" not in os.environ:
  os.environ["JIT_BATCH_SIZE"] = "0"
if "FLOAT16" not in os.environ:
  os.environ["FLOAT16"] = "1"
if "OPT" not in os.environ:
  os.environ["OPT"] = "99"


KEEP_BUFFERS = set()
original_reduce = Buffer.__reduce__
def stripped_reduce(self):
  if id(self) in KEEP_BUFFERS:
    return original_reduce(self)
  return (self.__class__, (self.device, self.size, self.dtype))
Buffer.__reduce__ = stripped_reduce


def compile_model(onnx_path, output_path, input_shapes=None, input_types=None):
  print(f"Compiling {onnx_path} -> {output_path}")
  run_onnx = OnnxRunner(onnx_path)

  if input_shapes is None:
    input_shapes = {name: spec.shape for name, spec in run_onnx.graph_inputs.items()}
  if input_types is None:
    input_types = {name: spec.dtype for name, spec in run_onnx.graph_inputs.items()}

  Tensor.manual_seed(100)
  inputs = {k: Tensor(Tensor.randn(*shp, dtype=input_types[k]).mul(8).realize().numpy(), device='NPY') for k, shp in sorted(input_shapes.items())}
  inputs = {k: v.to(Device.DEFAULT).realize() for k, v in inputs.items()}
  print(f"Realized all {len(inputs)} inputs on {Device.DEFAULT}")

  input_buf_ids = set()
  for _, v in inputs.items():
    if hasattr(v, '_buffer'):
      try:
        b = v._buffer()
        if b is not None:
          input_buf_ids.add(id(b))
      except Exception:
        pass

  if "vision" in onnx_path:
    onnx_jit = TinyJit(lambda **kwargs: next(iter(run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values())).cast('float32'), prune=True)
  else:
    onnx_jit = TinyJit(lambda **kwargs: [x.cast('float32').contiguous().realize() for x in run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values()], prune=True)

  for i in range(3):
    with Context(DEBUG=max(int(os.getenv("DEBUG", 0)), 2 if i == 2 else 1)):
      res = onnx_jit(**inputs)
      if isinstance(res, list):
        for x in res:
          x.numpy()
      else:
        res.numpy()
  print(f"Captured {len(onnx_jit.captured.jit_cache)} kernels")

  all_read_ids = set()
  all_written_ids = set()

  for ji in onnx_jit.captured.jit_cache:
    if len(ji.bufs) > 0:
      if ji.bufs[0] is not None:
        all_written_ids.add(id(ji.bufs[0]))

      for b in ji.bufs[1:]:
        if b is not None:
          all_read_ids.add(id(b))

  weight_candidates = all_read_ids - all_written_ids
  weight_ids = weight_candidates - input_buf_ids
  print(f"Identified {len(weight_ids)} weight candidates (Read-Only & Not Input).")
  total_weight_size = 0

  marked_count = 0
  for ji in onnx_jit.captured.jit_cache:
    for b in ji.bufs:
      if b is not None and id(b) in weight_ids:
        if id(b) not in KEEP_BUFFERS:
          KEEP_BUFFERS.add(id(b))
          total_weight_size += b.size * b.dtype.itemsize
          marked_count += 1

  print(f"Preserving {marked_count} unique weight buffers.")
  print(f"Total Preserved Weight Data Size: {total_weight_size / 1e6:.2f} MB")

  with open(output_path, "wb") as f:
    pickle.dump(onnx_jit, f)
  print(f"Saved {output_path}, pkl size: {os.path.getsize(output_path)/1e6:.2f} MB")

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python compile_split_tinygrad.py <input_onnx> <output_pkl>")
    sys.exit(1)

  input_onnx = sys.argv[1]
  output_pkl = sys.argv[2]
  compile_model(input_onnx, output_pkl)
