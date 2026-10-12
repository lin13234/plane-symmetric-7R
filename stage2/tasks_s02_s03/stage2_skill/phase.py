"""与机器人实测关节角解耦的显式技能相位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class PhaseSpec:
  """描述一个技能的相位运动规律。

  Attributes:
      mode: ``rotary`` 为连续转动，``inchworm`` 为上下限间往返。
      speed: 相位标量速度，单位 rad/s，必须为正数。
      limit: 尺蠖模式的对称相位上限，单位 rad。
      offset: episode 起点对应的相位偏置，单位 rad。
  """

  mode: Literal["rotary", "inchworm"]
  speed: float
  limit: float | None = None
  offset: float = 0.0

  def __post_init__(self) -> None:
    if self.speed <= 0.0:
      raise ValueError("phase speed must be positive")
    if self.mode == "inchworm" and (self.limit is None or self.limit <= 0.0):
      raise ValueError("inchworm phase requires a positive limit")
    if self.mode == "rotary" and self.limit is not None:
      raise ValueError("rotary phase must not define a limit")


def _as_control_tensor(value, like: torch.Tensor) -> torch.Tensor:
  """把标量或逐环境控制量转换为与 ``like`` 相同的张量。"""
  tensor = torch.as_tensor(value, device=like.device, dtype=like.dtype)
  return torch.broadcast_to(tensor, like.shape)


def phase_from_elapsed(
  elapsed: torch.Tensor,
  spec: PhaseSpec,
  *,
  direction=1.0,
  speed_scale=1.0,
  stopped=False,
) -> tuple[torch.Tensor, torch.Tensor]:
  """由独立经过时间和运行命令计算相位及相位速度。

  ``elapsed`` 是技能时钟，不读取机器人关节状态。``direction``、
  ``speed_scale`` 和 ``stopped`` 可以是标量，也可以是逐环境张量。
  """
  if elapsed.ndim != 1:
    raise ValueError("elapsed must be a one-dimensional tensor")
  if torch.any(elapsed < 0.0):
    raise ValueError("elapsed time must be non-negative")

  direction_t = _as_control_tensor(direction, elapsed)
  speed_scale_t = _as_control_tensor(speed_scale, elapsed)
  stopped_t = torch.broadcast_to(
    torch.as_tensor(stopped, device=elapsed.device, dtype=torch.bool),
    elapsed.shape,
  )
  if torch.any((direction_t != -1.0) & (direction_t != 1.0)):
    raise ValueError("direction must contain only -1 or +1")
  if torch.any(speed_scale_t <= 0.0):
    raise ValueError("speed_scale must be positive")

  effective_rate = float(spec.speed) * direction_t * speed_scale_t
  effective_rate = torch.where(
    stopped_t, torch.zeros_like(effective_rate), effective_rate
  )
  progress = effective_rate * elapsed

  if spec.mode == "rotary":
    return float(spec.offset) + progress, effective_rate

  # 三角波对正、负相位推进均有效。导数必须乘实际 signed rate，
  # 不能只根据三角波当前区间返回固定的正 speed。
  limit = float(spec.limit)
  unfolded = torch.remainder(progress + limit, 4.0 * limit)
  centered = limit - torch.abs(unfolded - 2.0 * limit)
  phase = float(spec.offset) + centered
  slope = torch.where(
    unfolded < 2.0 * limit,
    torch.ones_like(effective_rate),
    -torch.ones_like(effective_rate),
  )
  return phase, slope * effective_rate


def phase_state(
  step_count: torch.Tensor,
  step_dt: float,
  spec: PhaseSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
  """由episode步数计算相位和相位速度。

  这里故意不读取任何关节位置。这样，即使机器人受到接触阻碍，相位时钟仍然
  按指令推进，策略无法通过停住驱动关节来冻结参考轨迹。
  """

  if step_dt <= 0.0:
    raise ValueError("step_dt must be positive")

  time = step_count.to(dtype=torch.get_default_dtype()) * step_dt
  return phase_from_elapsed(time, spec)
