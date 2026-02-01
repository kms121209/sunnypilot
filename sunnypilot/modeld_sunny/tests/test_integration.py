import pytest
import numpy as np
from cereal import car
import cereal.messaging as messaging
from msgq.visionipc import VisionIpcServer, VisionIpcClient, VisionStreamType
from sunnypilot.modeld_sunny.modeld import AlpamayoModelD
from sunnypilot.modeld_v2.models.commonmodel_pyx import CLContext
from sunnypilot.modeld_sunny.fill_model_msg import fill_alpamayo_msg


@pytest.fixture(scope="module")
def cl_context():
  return CLContext()


@pytest.fixture(scope="module")
def modeld(cl_context):
  print("Initializing AlpamayoModelD...")
  return AlpamayoModelD(cl_context)


@pytest.fixture(scope="function")
def vipc_server():
  server_name = "camerad_test"
  server = VisionIpcServer(server_name)
  server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 1, 1024, 512)
  server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 1, 1024, 512)
  server.start_listener()
  yield server


def test_modeld(cl_context, modeld, vipc_server):
  v_ego = 20.0
  inputs = {
    'input_ids': np.zeros((1, 16), dtype=np.int64),
    'history': np.zeros((1, 16, 12), dtype=np.float32),
    'logic_pulse': modeld.logic_pulse,
    'traffic_convention': np.array([[1.0, 0.0]], dtype=np.float32),
    'v_ego': v_ego
  }

  server_name = "camerad_test"
  client_road = VisionIpcClient(server_name, VisionStreamType.VISION_STREAM_ROAD, False, cl_context)
  assert client_road.connect(True), "Road client failed to connect"
  client_wide = VisionIpcClient(server_name, VisionStreamType.VISION_STREAM_WIDE_ROAD, False, cl_context)
  assert client_wide.connect(True), "Wide client failed to connect"

  # NV12 size for 1024x512 = 1024*512 * 1.5 = 786432
  yuv_data = b'\x00' * 786432
  vipc_server.send(VisionStreamType.VISION_STREAM_ROAD, yuv_data)
  vipc_server.send(VisionStreamType.VISION_STREAM_WIDE_ROAD, yuv_data)
  buf_road = client_road.recv()
  buf_wide = client_wide.recv()
  assert buf_road is not None
  assert buf_wide is not None

  bufs = {'road': buf_road, 'wide': buf_wide}
  transforms = {'road': np.eye(3, dtype=np.float32), 'wide': np.eye(3, dtype=np.float32)}
  outputs = modeld.run(bufs, transforms, inputs, False)

  assert outputs is not None
  assert outputs["position"].shape == (1, 64, 3)
  assert outputs["velocity"].shape == (1, 64, 3)
  assert outputs["acceleration"].shape == (1, 64, 3)
  assert outputs["orientation"].shape == (1, 64, 3)
  assert "pred_pulse" in outputs
  assert "pred_light" in outputs
  assert "pred_lead" in outputs

  assert np.all(np.isfinite(outputs["position"])), "Position contains NaN/Inf"
  assert np.all(np.isfinite(outputs["velocity"])), "Velocity contains NaN/Inf"
  assert "consistency_error" in outputs
  assert outputs["consistency_error"] >= 0.0

  model = messaging.new_message('modelV2')
  CP = car.CarParams.new_message()
  CP.longitudinalActuatorDelay = 0.2
  fill_alpamayo_msg(model.modelV2, outputs, 12345, 0.0, 1e9, CP, 0.1, v_ego)
  # these just ensure that the model should outputs same action for the same black pixels
  assert model.modelV2.action.desiredAcceleration == pytest.approx(-6.75, abs=1e-2)
  assert model.modelV2.action.desiredCurvature == pytest.approx(-0.05, abs=1e-2)
  assert not model.modelV2.action.shouldStop
  assert model.modelV2.frameId == 12345
