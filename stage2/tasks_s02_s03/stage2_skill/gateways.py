"""G1 名义路径数据校验和有界确定性构型控制。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class MissingGatewayPathError(RuntimeError):
  """尚未提供来自机构学分析的 G1 名义路径。"""


@dataclass(frozen=True, slots=True)
class GatewayPath:
  """论文关节顺序 J1..J7 下的 G1 离散名义路径。"""

  waypoints: torch.Tensor

  @classmethod
  def from_waypoints(cls, waypoints: torch.Tensor | None) -> "GatewayPath":
    """校验路径，禁止用缺失值或非有限数值启动切换。"""
    if waypoints is None:
      raise MissingGatewayPathError(
        "缺少经过机构学与碰撞验证的 G1 名义路径，禁止构造过渡控制器"
      )
    tensor = torch.as_tensor(waypoints)
    if tensor.ndim != 2 or tensor.shape[1] != 7:
      raise ValueError("G1 path waypoints must contain seven joints")
    if tensor.shape[0] < 2:
      raise ValueError("G1 path must contain at least two waypoints")
    if not torch.all(torch.isfinite(tensor)):
      raise ValueError("G1 path waypoints must be finite")
    return cls(waypoints=tensor.detach().clone())


def bounded_configuration_target(
  current: torch.Tensor,
  desired: torch.Tensor,
  *,
  max_delta: float,
) -> torch.Tensor:
  """生成单控制步有界的位置目标，避免切换时关节目标突跳。"""
  if current.shape != desired.shape:
    raise ValueError("current and desired configurations must have equal shape")
  if max_delta <= 0.0:
    raise ValueError("max_delta must be positive")
  correction = torch.clamp(desired - current, -max_delta, max_delta)
  return current + correction


class ApproachGatewayG1Controller:
  """沿已验证离散路径逐点接近 G1 的确定性控制器。"""

  def __init__(self, path: GatewayPath, *, max_delta: float = 0.05) -> None:
    self.path = path
    if max_delta <= 0.0:
      raise ValueError("max_delta must be positive")
    self.max_delta = float(max_delta)

  def target(self, current: torch.Tensor, waypoint_index: int) -> torch.Tensor:
    """返回指定路径点对应的有界七关节目标。"""
    if current.shape[-1] != 7:
      raise ValueError("current configuration must contain seven joints")
    if not 0 <= waypoint_index < self.path.waypoints.shape[0]:
      raise IndexError("waypoint_index outside G1 path")
    desired = self.path.waypoints[waypoint_index].to(
      device=current.device, dtype=current.dtype
    )
    desired = torch.broadcast_to(desired, current.shape)
    return bounded_configuration_target(current, desired, max_delta=self.max_delta)


class HoldBranchConfigurationController:
  """切换前后使用的有界构型保持器。"""

  def __init__(self, desired: torch.Tensor, *, max_delta: float = 0.05) -> None:
    self.desired = torch.as_tensor(desired).detach().clone()
    self.max_delta = float(max_delta)

  def target(self, current: torch.Tensor) -> torch.Tensor:
    desired = self.desired.to(device=current.device, dtype=current.dtype)
    desired = torch.broadcast_to(desired, current.shape)
    return bounded_configuration_target(current, desired, max_delta=self.max_delta)
