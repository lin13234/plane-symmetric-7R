"""Stage2 名义控制与强化学习残差的组合规则。"""

from __future__ import annotations

import torch


def compose_residual_target(
  nominal_target: torch.Tensor,
  policy_residual: torch.Tensor,
  residual_scale: float,
) -> torch.Tensor:
  """生成最终四电机位置目标。

  策略动作先限制到 [-1, 1]，再乘残差尺度。该顺序保证神经网络输出无论
  多大，都不能突破物理上定义的残差幅值边界。
  """

  if nominal_target.shape != policy_residual.shape:
    raise ValueError("nominal target and policy residual must have identical shapes")
  if residual_scale < 0.0:
    raise ValueError("residual_scale must be non-negative")

  bounded_residual = torch.clamp(policy_residual, min=-1.0, max=1.0)
  return nominal_target + residual_scale * bounded_residual
