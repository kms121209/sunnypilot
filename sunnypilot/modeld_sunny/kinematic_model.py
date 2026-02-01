from tinygrad.tensor import Tensor


def action_to_traj(action: Tensor, v0: Tensor, dt: float = 0.1):
  """
  This function is a lightweight tinygrad transformation of the unicycle accel physics model based on Nvidia's
  unicycle model https://github.com/NVlabs/alpamayo/blob/main/src/alpamayo_r1/action_space/unicycle_accel_curvature.py

  Integrate action (accel, kappa) to trajectory (x, y, theta)
  Args:
    action: (B, T, 2) [accel, kappa]
    v0: (B,) Initial velocity
    dt: Time step
  Returns:
    res: Dict containing position, velocity, acceleration, orientation, orientation_rate
  """
  B, T, _ = action.shape
  ACCEL_MEAN = 0.02902695
  ACCEL_STD = 0.68104267
  CURV_MEAN = 0.00026922
  CURV_STD = 0.02614828

  accel = action[..., 0] * ACCEL_STD + ACCEL_MEAN
  kappa = action[..., 1] * CURV_STD + CURV_MEAN

  # v_{t+1} = v_t + a_t * dt
  v_diff = accel * dt
  v_seq = v_diff.cumsum(axis=1) + v0.reshape(B, 1)   # cumulative sum over T dimension (axis 1)
  velocity = v0.reshape(B, 1).cat(v_seq, dim=1)

  # theta_{t+1} = theta_t + kappa_t * (v_t * dt + 0.5 * a_t * dt^2)
  dt_2_term = 0.5 * (dt**2)
  dtheta = kappa * (velocity[:, :-1] * dt + accel * dt_2_term)
  theta_seq = dtheta.cumsum(axis=1)
  theta = Tensor.zeros(B, 1, device=action.device, dtype=action.dtype).cat(theta_seq, dim=1)

  # trapezoidal euler
  half_dt = 0.5 * dt
  v_cos = velocity * theta.cos()
  v_sin = velocity * theta.sin()

  dx = (v_cos[:, :-1] + v_cos[:, 1:]) * half_dt
  dy = (v_sin[:, :-1] + v_sin[:, 1:]) * half_dt
  x = dx.cumsum(axis=1)
  y = dy.cumsum(axis=1)

  res = {}
  res['action'] = accel.stack(kappa, dim=-1)  # raw model output

  # (x, y, 0)
  res['position'] = x.stack(y, Tensor.zeros(B, T, device=action.device, dtype=action.dtype), dim=-1)
  # (vx, vy, 0)
  res['velocity'] = v_cos[:, 1:].stack(v_sin[:, 1:], Tensor.zeros(B, T, device=action.device, dtype=action.dtype), dim=-1)
  # ax = accel * cos(theta), ay = accel * sin(theta), 0
  res['acceleration'] = (accel * theta[:, 1:].cos()).stack(accel * theta[:, 1:].sin(), Tensor.zeros(B, T, device=action.device, dtype=action.dtype), dim=-1)
  # (0, 0, theta)
  res['orientation'] = Tensor.zeros(B, T, device=action.device, dtype=action.dtype).stack(Tensor.zeros(B, T, device=action.device, dtype=action.dtype), theta[:, 1:], dim=-1)
  # (0, 0, dtheta/dt)
  res['orientation_rate'] = Tensor.zeros(B, T, device=action.device, dtype=action.dtype).stack(Tensor.zeros(B, T, device=action.device, dtype=action.dtype), dtheta / dt, dim=-1)
  return res
