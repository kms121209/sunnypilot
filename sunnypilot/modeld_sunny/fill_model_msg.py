import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, get_curvature_from_plan


def interp_vec(t_out, t_in, vec):
  # vec shape (N, 3), output (M, 3)
  return np.stack([np.interp(t_out, t_in, vec[:, i]) for i in range(3)], axis=1)


def fill_alpamayo_msg(modelV2, net_outputs, frame_id, frame_drop_ratio, timestamp_eof, CP, lat_delay, v_ego):
  modelV2.frameId = frame_id
  modelV2.frameIdExtra = frame_id
  modelV2.timestampEof = timestamp_eof
  modelV2.frameDropPerc = frame_drop_ratio * 100.0

  modelV2.init('laneLines', 4)
  modelV2.init('roadEdges', 2)
  modelV2.init('laneLineProbs', 4)
  modelV2.init('roadEdgeStds', 2)

  for i in range(4):
    l = modelV2.laneLines[i]
    l.t = [0.0]
    l.x = [0.0]
    l.y = [0.0]
    l.z = [0.0]
    modelV2.laneLineProbs[i] = 0.0

  for i in range(2):
    e = modelV2.roadEdges[i]
    e.t = [0.0]
    e.x = [0.0]
    e.y = [0.0]
    e.z = [0.0]
    modelV2.roadEdgeStds[i] = 1.0


  leads = modelV2.init('leadsV3', 1)
  lead = leads[0]
  pred_lead = net_outputs['pred_lead'][0]
  prob_logit = float(pred_lead[0])
  dist_pred = float(pred_lead[1] * 100.0)
  dist_sigma = float(np.exp(pred_lead[2]))

  v_rel_pred = float(pred_lead[3])
  v_sigma = float(np.exp(pred_lead[4]))

  a_rel_pred = float(pred_lead[5])
  a_sigma = float(np.exp(pred_lead[6]))

  prob = float(1.0 / (1.0 + np.exp(-prob_logit)))

  lead.prob = prob
  lead.probTime = 0.0

  # X(t) = X0 + V_rel*t + 0.5*A_rel*t^2
  T = ModelConstants.T_IDXS
  lead.t = list(T)
  lead.x = [float(dist_pred + v_rel_pred * t + 0.5 * a_rel_pred * t**2) for t in T]
  lead.v = [float(v_ego + v_rel_pred + a_rel_pred * t) for t in T]
  a_ego = net_outputs['acceleration'][0, 0, 0] # T=0 ego accel estimate (x component)
  lead.a = [float(a_ego + a_rel_pred)] * len(T)
  lead.y = [0.0] * len(T)

  lead.xStd = [max(0.5, dist_sigma * 100.0)] * len(T)
  lead.yStd = [1.0] * len(T)
  lead.vStd = [max(0.1, v_sigma)] * len(T)
  lead.aStd = [max(0.1, a_sigma)] * len(T)

  modelV2.meta.engagedProb = 1.0
  desire_pred = [0.0] * 8

  if 'pred_light' in net_outputs:
    red_prob = float(1.0 / (1.0 + np.exp(-net_outputs['pred_light'][0, 1] + net_outputs['pred_light'][0, 0])))
    desire_pred[4] = red_prob

  modelV2.meta.desirePrediction = desire_pred
  modelV2.meta.desireState = [0.0] * 8
  reasoning_error = net_outputs.get('consistency_error', 0.0)

  if reasoning_error < 0.5:
    modelV2.confidence = "green"
  elif reasoning_error < 1.5:
    modelV2.confidence = "yellow"
  else:
    modelV2.confidence = "red"

  ALPAMAYO_T_IDXS = np.arange(1, 65) * 0.1 # 64 steps at .1s intervals
  t_idxs = ModelConstants.T_IDXS
  t_all = np.concatenate(([0.0], ALPAMAYO_T_IDXS)) # this model starts at t=0.1 so if we prepend 0.0 and interpolate for t=now it should match op

  pos_interp = interp_vec(t_idxs, t_all, np.vstack((np.zeros(3), net_outputs['position'][0])))
  pos_std_interp = interp_vec(t_idxs, t_all, np.vstack((np.zeros(3), net_outputs.get('position_std', np.ones((64, 3)) * 0.1))))
  vel_interp = interp_vec(t_idxs, t_all, np.vstack(([v_ego, 0.0, 0.0], net_outputs['velocity'][0])))
  acc_interp = interp_vec(t_idxs, t_all, np.vstack((net_outputs['acceleration'][0][0], net_outputs['acceleration'][0])))
  rot_interp = interp_vec(t_idxs, t_all, np.vstack((np.zeros(3), net_outputs['orientation'][0])))
  rate_interp = interp_vec(t_idxs, t_all, np.vstack((net_outputs['orientation_rate'][0][0], net_outputs['orientation_rate'][0])))

  # https://www.mathworks.com/help/vdynblks/ug/coordinate-systems-in-vehicle-dynamics-blockset.html
  # following SAE J670 and ISO 8855, for sunnymayo model: x is forward (f), y is left (lat), z is up/vert
  # Openpilot Modelv2 and camerad expects SAE J670: x is forward, y is right, z is down

  modelV2.position.t = t_idxs  # time, obviously
  modelV2.position.x = pos_interp[:, 0].tolist()  # f dist
  modelV2.position.y = (-pos_interp[:, 1]).tolist()  # lat offset (Flip L->R)
  modelV2.position.z = (-pos_interp[:, 2]).tolist()  # vert offset (Flip U->D) (elevation)
  modelV2.position.xStd = pos_std_interp[:, 0].tolist()
  modelV2.position.yStd = pos_std_interp[:, 1].tolist()
  modelV2.position.zStd = pos_std_interp[:, 2].tolist()

  modelV2.velocity.t = t_idxs
  modelV2.velocity.x = vel_interp[:, 0].tolist()  # f vel (vego)
  modelV2.velocity.y = (-vel_interp[:, 1]).tolist()  # lat vel (curvature)
  modelV2.velocity.z = (-vel_interp[:, 2]).tolist()  # vert vel

  modelV2.acceleration.t = t_idxs
  modelV2.acceleration.x = acc_interp[:, 0].tolist()  # f accel  (aego)
  modelV2.acceleration.y = (-acc_interp[:, 1]).tolist()  # lat accel
  modelV2.acceleration.z = (-acc_interp[:, 2]).tolist()  # vert accel

  modelV2.orientation.t = t_idxs
  modelV2.orientation.x = rot_interp[:, 0].tolist()  # roll (treated as 0)
  modelV2.orientation.y = (-rot_interp[:, 1]).tolist()  # pitch (from z-slope)
  modelV2.orientation.z = (-rot_interp[:, 2]).tolist()  # yaw  (heading)

  modelV2.orientationRate.t = t_idxs
  modelV2.orientationRate.x = rate_interp[:, 0].tolist()  # roll rate
  modelV2.orientationRate.y = (-rate_interp[:, 1]).tolist()  # pitch rate (Flip U->D)
  modelV2.orientationRate.z = (-rate_interp[:, 2]).tolist()  # yaw rate (Flip L->R)

  long_action_t = CP.longitudinalActuatorDelay + DT_MDL
  desired_accel, should_stop = get_accel_from_plan(vel_interp[:, 0], acc_interp[:, 0], t_idxs, action_t=long_action_t)
  modelV2.action.desiredAcceleration = float(desired_accel)
  modelV2.action.shouldStop = bool(should_stop)

  lat_action_t = lat_delay + DT_MDL
  desired_curvature = get_curvature_from_plan(-rot_interp[:, 2], -rate_interp[:, 2], t_idxs, vego=v_ego, action_t=lat_action_t)
  modelV2.action.desiredCurvature = float(desired_curvature)


def fill_pose_msg(camera_odometry, net_outputs, frame_id, timestamp_eof):
  camera_odometry.frameId = frame_id
  camera_odometry.timestampEof = timestamp_eof

  trans = net_outputs['velocity'][0, 0].copy()
  trans[1] *= -1.0
  trans[2] *= -1.0
  camera_odometry.trans = trans.tolist()

  std_val = float(max(0.01, net_outputs.get('consistency_error', 0.1)))
  camera_odometry.transStd = [std_val, std_val, std_val]
  rot = net_outputs['orientation_rate'][0, 0].copy()
  rot[1] *= -1.0
  rot[2] *= -1.0
  camera_odometry.rot = rot.tolist()

  rot_std = std_val * 0.1
  camera_odometry.rotStd = [rot_std, rot_std, rot_std]
