import time
import numpy as np
import os
import platform
from setproctitle import setproctitle
from openpilot.system.hardware import TICI

if TICI:
  os.environ['DEV'] = 'QCOM'
elif platform.system() == "Darwin":
  os.environ['DEV'] = "METAL"
else:
  os.environ['DEV'] = 'CPU'
USBGPU = "USBGPU" in os.environ
if USBGPU:
  os.environ['DEV'] = 'AMD'
  os.environ['AMD_IFACE'] = 'USB'

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

import cereal.messaging as messaging
from cereal import car
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.transformations.model import bigmodel_frame_from_calib_frame
from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.common.realtime import Ratekeeper
from openpilot.selfdrive.modeld.runners.tinygrad_helpers import qcom_tensor_from_opencl_address

from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.modeld_v2.models.commonmodel_pyx import DrivingModelFrame, CLContext
from openpilot.sunnypilot.modeld_sunny.kinematic_model import action_to_traj
from openpilot.sunnypilot.modeld_v2.camera_offset_helper import CameraOffsetHelper
from openpilot.sunnypilot.modeld_sunny.loader import load_compiled_model
from openpilot.sunnypilot.modeld_sunny.input_id_helper import InputIDHelper
from openpilot.sunnypilot.modeld_sunny.fill_model_msg import fill_alpamayo_msg, fill_pose_msg

PROCESS_NAME = "selfdrive.modeld.openpilot.sunnypilot.modeld_sunny"


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


def safe_exp(x):
  return np.exp(np.clip(x, -np.inf, 11))


def softmax(x, axis=-1):
  x = x - np.max(x, axis=axis, keepdims=True)
  x = safe_exp(x)
  return x / np.sum(x, axis=axis, keepdims=True)


class AlpamayoModelD:
  def __init__(self, context: CLContext):
    self.params = Params()
    self.context = context

    self.model_vision = load_compiled_model("student_vision")
    self.model_policy = load_compiled_model("student_policy")
    self.model_loaded = self.model_vision is not None and self.model_policy is not None

    self.vision_input_names = ['road', 'wide']
    self.vision_input_shapes = {
      'road': (1, 3, 512, 1024),
      'wide': (1, 3, 512, 1024)
    }

    self.frames = {name: DrivingModelFrame(context, 1024, 512, buffer_length=4) for name in self.vision_input_names}
    self.history_buffer = np.zeros((16, 3), dtype=np.float32)
    self.logic_pulse = np.zeros((1, 2048), dtype=np.float32)

  def run(self, bufs, transforms, inputs, prepare_only):
    if prepare_only:
      return None
    if not hasattr(self, 'vision_inputs'):
      self.vision_inputs = {}

    imgs_cl = {n: self.frames[n].prepare(bufs[n], transforms[n].flatten()) for n in self.vision_input_names if bufs.get(n)}

    if TICI and not USBGPU:
      for k, v in imgs_cl.items():
        if k not in self.vision_inputs:
          self.vision_inputs[k] = qcom_tensor_from_opencl_address(v.mem_address, self.vision_input_shapes[k], dtype=dtypes.uint8)
    else:
      for k, v in imgs_cl.items():
        self.vision_inputs[k] = Tensor(self.frames[k].buffer_from_cl(v).reshape(self.vision_input_shapes[k]), dtype=dtypes.uint8).realize()

    img_t = Tensor.stack([self.vision_inputs['wide'].cast(dtypes.float32) / 255.0,
                          self.vision_inputs['road'].cast(dtypes.float32) / 255.0], dim=1).unsqueeze(0)

    vis_res = self.model_vision(
      history=Tensor(inputs["history"]).contiguous().realize(),
      img=img_t.contiguous().realize(),
      input_ids=Tensor(inputs["input_ids"]).contiguous().realize(),
      logic_pulse=Tensor(inputs["logic_pulse"]).contiguous().realize()
    )
    context = vis_res.contiguous().realize()

    x_input = Tensor.zeros(1, 64, 2, device=os.environ.get("DEV"), dtype=dtypes.float32)
    v_mu, v_std, pred_pulse, state_mu, state_std, pred_light, pred_lead, hypot_logits = self.model_policy(
      context=context,
      noisy_action=x_input.contiguous().realize(),
      t=Tensor(np.array([[0.0]], dtype=np.float32)).contiguous().realize(), # t=0
      traffic=Tensor(inputs["traffic_convention"]).contiguous().realize()
    )

    weights = softmax(hypot_logits.numpy(), axis=1) # (B, M)
    winner_idx = np.argmax(weights[0])

    v_winner = v_mu[:, winner_idx]
    state_winner = state_mu[:, winner_idx]
    state_std_winner = state_std[:, winner_idx]

    outputs_tg = action_to_traj(v_winner, Tensor([inputs["v_ego"]], dtype=dtypes.float32), dt=0.1)
    outputs = {k: v.numpy() for k, v in outputs_tg.items()}
    outputs.update({
      "pred_pulse": pred_pulse.numpy(),
      "pred_light": pred_light[0:1].numpy(),
      "pred_lead": pred_lead[0:1].numpy(),
      "weights": weights[0]
    })

    # Inject world positions for Z/Pitch
    pos_world = state_winner[0].numpy()
    pos_std = np.exp(state_std_winner[0].numpy())
    outputs["position"][0, :, 2] = pos_world[:, 2]
    outputs["position_std"] = pos_std # log_sigma -> sigma
    d_pos = np.diff(outputs["position"][0], axis=0, prepend=np.zeros((1, 3)))
    d_dist = np.maximum(np.linalg.norm(d_pos[:, :2], axis=1), 1e-4)
    pitch = np.arctan2(np.diff(pos_world[:, 2], prepend=0.0), d_dist)
    outputs["orientation"][0, :, 1] = pitch
    outputs["orientation_rate"][0, :, 1] = np.diff(pitch, prepend=0.0) / 0.1
    outputs["velocity"][0, :, 2] = np.linalg.norm(outputs["velocity"][0, :, :2], axis=1) * np.tan(pitch)
    outputs["consistency_error"] = float(np.mean(np.linalg.norm(outputs["position"][0] - pos_world, axis=1)))

    return outputs


def main():
  setproctitle(PROCESS_NAME)
  config_realtime_process(7, 54)
  # Loop runs at 20Hz to match camera acquisition.
  # Model inference runs at 10Hz via frame skipping.
  rk = Ratekeeper(1.0 / DT_MDL)
  cl_context = CLContext()
  modeld = AlpamayoModelD(cl_context)

  # Load CarParams
  cloudlog.warning("Modeld: Waiting for CarParams...")
  CP = messaging.log_from_bytes(Params().get("CarParams", block=True), car.CarParams)
  cloudlog.warning("Modeld: Got CarParams")

  camera_offset_helper = CameraOffsetHelper()
  input_id_helper = InputIDHelper()

  if modeld.model_loaded:
    cloudlog.warning("Modeld: Successfully loaded compiled student model.")

  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "liveDelay", "livePose", "driverMonitoringState"])

  # VisionIPC Clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()

  # filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / 20.0)
  last_vipc_frame_id = 0
  run_count = 0
  lat_delay = 0.0
  live_calib_seen = False

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
          extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    # 10Hz Execution Check (Skip odd frames)
    # We use main camera frameId as the clock
    if meta_main.frame_id % 2 != 0:
      last_vipc_frame_id = meta_main.frame_id
      continue

    sm.update(0)
    v_ego = sm['carState'].vEgo if sm.seen['carState'] else 0.0

    yaw_rate = 0.0
    if sm.seen['livePose'] and sm['livePose'].angularVelocityDevice.valid:
      yaw_rate = sm['livePose'].angularVelocityDevice.z

    if sm.frame % 60 == 0:
      lat_delay = get_lat_delay(modeld.params, sm["liveDelay"].lateralDelay)
      camera_offset_helper.set_offset(modeld.params.get("CameraOffset", return_default=True))

    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      calib_from_bigmodel = np.linalg.inv(bigmodel_frame_from_calib_frame[:, :3])
      device_from_calib = rot_from_euler(device_from_calib_euler)
      camera_from_calib_main = (dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics) @ view_frame_from_device_frame @ device_from_calib
      model_transform_main = camera_from_calib_main @ calib_from_bigmodel
      camera_from_calib_extra = dc.ecam.intrinsics @ view_frame_from_device_frame @ device_from_calib
      model_transform_extra = camera_from_calib_extra @ calib_from_bigmodel

      model_transform_main, model_transform_extra = camera_offset_helper.update(model_transform_main, model_transform_extra, sm, main_wide_camera)
      live_calib_seen = True

    # Track dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

    bufs = {'road': buf_main, 'wide': buf_extra}
    transforms = {'road': model_transform_main, 'wide': model_transform_extra}

    dt = 0.1
    d_yaw = yaw_rate * dt
    d_pos = v_ego * dt * np.array([np.cos(d_yaw/2), np.sin(d_yaw/2)])
    rot = np.array([[np.cos(-d_yaw), -np.sin(-d_yaw)], [np.sin(-d_yaw), np.cos(-d_yaw)]])
    modeld.history_buffer[:, :2] = (modeld.history_buffer[:, :2] - d_pos) @ rot.T
    modeld.history_buffer[:, 2] -= d_yaw
    modeld.history_buffer = np.roll(modeld.history_buffer, -1, axis=0)
    modeld.history_buffer[-1] = 0.

    hist_input = modeld.history_buffer.copy()
    hist_input[:, 1] *= -1.0
    hist_input[:, 2] *= -1.0
    yaws_fixed = hist_input[:, 2]
    cos_y, sin_y = np.cos(yaws_fixed), np.sin(yaws_fixed)
    zeros, ones = np.zeros_like(cos_y), np.ones_like(cos_y)
    rot_flat = np.column_stack([cos_y, -sin_y, zeros, sin_y, cos_y, zeros, zeros, zeros, ones])

    inputs = {
      'input_ids': input_id_helper.update(sm),
      'history': np.column_stack([hist_input[:, :2], zeros, rot_flat])[None, ...].astype(np.float32),
      'logic_pulse': modeld.logic_pulse,
      'traffic_convention': np.array([[0.0, 1.0]] if sm["driverMonitoringState"].isRHD else [[1.0, 0.0]], dtype=np.float32),
      'v_ego': v_ego
    }

    t0 = time.monotonic()
    outputs = modeld.run(bufs, transforms, inputs, prepare_only)
    t1 = time.monotonic()
    if not prepare_only:
      cloudlog.warning(f"Modeld: Inference took {(t1-t0)*1000:.2f} ms")
    last_vipc_frame_id = meta_main.frame_id

    if outputs is not None:
      modeld.logic_pulse[:] = outputs["pred_pulse"]
      model_msg = messaging.new_message('modelV2')
      drivingdata_msg = messaging.new_message('drivingModelData')
      posenet_msg = messaging.new_message('cameraOdometry')

      fill_alpamayo_msg(model_msg.modelV2, outputs, meta_main.frame_id, frame_drop_ratio, meta_main.timestamp_eof, CP, lat_delay, v_ego)
      model_msg.valid = live_calib_seen and (vipc_dropped_frames < 1)

      fill_pose_msg(posenet_msg.cameraOdometry, outputs, meta_main.frame_id, meta_main.timestamp_eof)
      posenet_msg.valid = live_calib_seen and (vipc_dropped_frames < 1)

      drivingdata_msg.drivingModelData.frameId = meta_main.frame_id

      pm.send('drivingModelData', drivingdata_msg)
      pm.send('cameraOdometry', posenet_msg)
      pm.send('modelV2', model_msg)
    rk.keep_time()


if __name__ == "__main__":
  main()
