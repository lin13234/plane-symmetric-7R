"""禁止实体杆件碰撞的连续接触保护。"""

from __future__ import annotations

import torch


class ForbiddenContactStreak:
  """对每个并行环境独立统计连续禁止接触的控制步数。"""

  def __init__(
    self,
    *,
    num_envs: int,
    threshold_steps: int = 10,
    device: str | torch.device,
  ) -> None:
    if threshold_steps < 1:
      raise ValueError("threshold_steps must be positive")
    self.threshold_steps = threshold_steps
    self.counts = torch.zeros(num_envs, dtype=torch.long, device=device)

  def update(self, forbidden_contact: torch.Tensor) -> torch.Tensor:
    """更新计数并返回本控制步应终止的环境掩码。"""
    if forbidden_contact.dtype is not torch.bool:
      forbidden_contact = forbidden_contact.to(torch.bool)
    self.counts = torch.where(
      forbidden_contact,
      self.counts + 1,
      torch.zeros_like(self.counts),
    )
    return self.counts >= self.threshold_steps

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """清零指定环境；None 表示全部环境。"""
    self.counts[env_ids] = 0
