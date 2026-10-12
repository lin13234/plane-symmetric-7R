"""分支切换训练的先决条件和成功判据。

本模块不会伪造 G1 路径。只有几何路径、完整碰撞检查和分支判定器均被
外部检测确认后，切换环境才允许进入正式训练。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .command import BranchId


@dataclass(frozen=True, slots=True)
class TransitionReadiness:
  """双向分支切换任务的三项硬性解锁条件。"""

  gateway_path_configured: bool
  collision_path_verified: bool
  branch_classifier_verified: bool

  @property
  def missing_requirements(self) -> tuple[str, ...]:
    missing: list[str] = []
    if not self.gateway_path_configured:
      missing.append("gateway_path_configured")
    if not self.collision_path_verified:
      missing.append("collision_path_verified")
    if not self.branch_classifier_verified:
      missing.append("branch_classifier_verified")
    return tuple(missing)

  @property
  def trainable(self) -> bool:
    return not self.missing_requirements

  def assert_trainable(self) -> None:
    """未满足任一条件时显式拒绝启动切换训练。"""
    if self.missing_requirements:
      missing = ", ".join(self.missing_requirements)
      raise RuntimeError(f"分支切换禁止训练；缺少前置验证: {missing}")


def consecutive_target_confirmation(
  labels: Sequence[BranchId | str],
  *,
  target: BranchId,
  required_frames: int,
) -> bool:
  """仅当序列末尾连续多帧属于目标分支时判定切换成功。"""
  if required_frames <= 0:
    raise ValueError("required_frames must be positive")
  if len(labels) < required_frames:
    return False
  tail = labels[-required_frames:]
  return all(BranchId(label) is target for label in tail)
