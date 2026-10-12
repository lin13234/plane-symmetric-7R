"""Stage2 统一回放指标：位移、周期数、速度、漂移与跟踪误差。

纯函数设计：输入回放轨迹与命令相位，输出与论文 Table 4 口径一致的
指标字典。评估器只负责采样数据，本模块负责计算，保证可单测、可复核。
横向漂移（lateral_drift_m / normalized_drift / motion_diagnostics）仅作诊断以
表征不同运动分支，不进入 success_criteria 失败判据；success_criteria 只含
周期数、每周期位移、前进速度、跟踪 RMSE 四项。
"""

from __future__ import annotations

import math

import numpy as np

# 桥接特征长度（m），归一化速度/周期位移用。
L_B = 0.06


def wrap_pi(x: np.ndarray) -> np.ndarray:
  return (x + np.pi) % (2.0 * np.pi) - np.pi


def count_cycles(
  command_phase: np.ndarray,
  cycle_type: str,
  inchworm_limit: float | None = None,
) -> float:
  """由命令相位计算完成的周期数。

  rotary  : unwrapped 相位增量 / 2π（连续旋转）；
  inchworm: 三角波相位路径总长 / (4·limit)。每个往返（0→+L→-L→0）的
      相位路径恒等于 4·limit，与回放是否走满一个往返无关；limit 来自
      PhaseSpec，而不是数据的实测摆幅（数据摆幅在未走满时会偏小）。
  """
  if cycle_type == "rotary":
    unwrapped = np.unwrap(command_phase)
    return float(abs(unwrapped[-1] - unwrapped[0]) / (2.0 * math.pi))
  if inchworm_limit is None or inchworm_limit <= 0.0:
    raise ValueError("inchworm cycle counting requires a positive limit")
  # 注意：三角波的折返是真实路径，不能 unwrap（unwrap 会把 ±L 折返
  # 误判为缠绕而抹掉 2L 的路径）。
  total_variation = float(np.abs(np.diff(command_phase)).sum())
  return max(total_variation / (4.0 * float(inchworm_limit)), 0.0)


def compute_replay_metrics(
  times: np.ndarray,
  root_xy: np.ndarray,
  command_phase: np.ndarray,
  joints_paper: np.ndarray,
  ref_paper: np.ndarray,
  cycle_type: str,
  inchworm_limit: float | None = None,
  segment_start: int = 0,
  forward_axis: np.ndarray | None = None,
  lateral_axis: np.ndarray | None = None,
  l_b: float = L_B,
) -> dict:
  """计算统一回放指标。

  Args:
      times: (N,) 每步时间。
      root_xy: (N, 2) 世界 XY。
      command_phase: (N,) 显式命令相位（与训练同源）。
      joints_paper: (N, 7) 实测 paper J1..J7。
      ref_paper: (N, 7) 名义参考（由命令相位计算）。
      cycle_type: "rotary" 或 "inchworm"。
      segment_start: 最后一个连续段的起点索引（reset 或落地结束后第一帧；
          无 reset 时取 0）。位移、周期与跟踪误差只在该段内计算，
          避免把 reset 跳变或落地瞬态计入指标。
      forward_axis: 前进轴单位向量（地面系），默认世界 -Y。
      lateral_axis: 横向轴单位向量，默认世界 +X。
  """
  if forward_axis is None:
    forward_axis = np.array([0.0, -1.0])
  if lateral_axis is None:
    lateral_axis = np.array([1.0, 0.0])

  seg = slice(segment_start, len(times))
  t_seg = times[seg]
  xy_seg = root_xy[seg]
  phase_seg = command_phase[seg]
  joints_seg = joints_paper[seg]
  ref_seg = ref_paper[seg]
  if len(t_seg) < 2:
    raise ValueError("segment too short to compute metrics")

  disp = xy_seg[-1] - xy_seg[0]
  delta_f = float(disp @ forward_axis)
  delta_l = float(disp @ lateral_axis)

  T = float(t_seg[-1] - t_seg[0])
  # 周期数同样只在连续段内计算，排除落地/reset 瞬态。
  n_cycles = count_cycles(phase_seg, cycle_type, inchworm_limit)
  d_cycle = delta_f / n_cycles if n_cycles > 1e-9 else 0.0
  v_f = delta_f / T if T > 1e-9 else 0.0
  d_l = float(abs(delta_l))
  d_l_norm = d_l / max(abs(delta_f), 1e-9)

  err = wrap_pi(joints_seg - ref_seg)
  rmse_q = float(np.sqrt(np.mean(err**2)))

  return {
    "horizon_steps": int(len(times)),
    "segment_start": int(segment_start),
    "T_seconds": T,
    "Delta_f_m": delta_f,
    "Delta_l_m": delta_l,
    "N_cycles": n_cycles,
    "d_cycle_m": d_cycle,
    "d_cycle_over_Lb": d_cycle / l_b,
    "mean_speed_m_s": v_f,
    "mean_speed_over_Lb_s": v_f / l_b,
    "lateral_drift_m": d_l,
    "normalized_drift": d_l_norm,
    "tracking_rmse_rad": rmse_q,
    "L_b": l_b,
    # 横向漂移仅作诊断（表征不同运动分支），不作为失败判据。
    "motion_diagnostics": {
      "lateral_drift_m": d_l,
      "normalized_lateral_drift": d_l_norm,
    },
    "success_criteria": {
      "N_c>=5": bool(n_cycles >= 5.0),
      "d_cycle/Lb>=0.02": bool(d_cycle / l_b >= 0.02),
      "v_f/Lb>=0.01": bool(v_f / l_b >= 0.01),
      "rmse_q<=0.25": bool(rmse_q <= 0.25),
    },
  }
