"""安全 Joint2 名义轨迹与四电机残差的 mjlab 动作项。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

from .reference import ACTUATED_JOINT_NAMES, joint2_reference_torch


@dataclass(kw_only=True)
class Joint2SafeResidualActionCfg(BaseActionCfg):
  """在审计窗口内往复的 Joint2 名义参考，加四维限幅策略残差。"""

  actuator_names: tuple[str, ...] = ACTUATED_JOINT_NAMES
  residual_scale: float = 0.05
  period_s: float = 4.0
  safe_lower: float = -1.492257
  safe_upper: float = 2.303835

  def __post_init__(self) -> None:
    self.transmission_type = TransmissionType.JOINT
    if self.residual_scale < 0.0:
      raise ValueError("residual_scale must be non-negative")
    if self.period_s <= 0.0 or self.safe_lower >= self.safe_upper:
      raise ValueError("invalid safe phase schedule")

  def build(self, env) -> "Joint2SafeResidualAction":
    return Joint2SafeResidualAction(self, env)


class Joint2SafeResidualAction(BaseAction):
  """每个控制步下发安全名义目标，策略只能施加有限幅值残差。"""

  cfg: Joint2SafeResidualActionCfg

  def __init__(self, cfg: Joint2SafeResidualActionCfg, env) -> None:
    super().__init__(cfg=cfg, env=env)
    if tuple(self.target_names) != ACTUATED_JOINT_NAMES:
      raise ValueError(f"actuator order mismatch: {tuple(self.target_names)}")
    self._bounded_residual = torch.zeros_like(self._raw_actions)
    self._phase = torch.zeros(self.num_envs, device=self.device)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions.to(self.device)
    self._bounded_residual[:] = torch.clamp(self._raw_actions, -1.0, 1.0)
    self._processed_actions[:] = self._bounded_residual

  def apply_actions(self) -> None:
    time_s = self._env.episode_length_buf.to(torch.float32) * self._env.step_dt
    cycle = torch.remainder(time_s, self.cfg.period_s) / self.cfg.period_s
    triangle = torch.where(cycle <= 0.5, 2.0 * cycle, 2.0 * (1.0 - cycle))
    self._phase[:] = self.cfg.safe_lower + triangle * (
      self.cfg.safe_upper - self.cfg.safe_lower
    )
    qpos = joint2_reference_torch(self._phase)
    nominal = qpos[:, (0, 1, 5, 6)]
    target = nominal + self._bounded_residual * self.cfg.residual_scale
    self._entity.set_joint_position_target(target, joint_ids=self.target_ids)

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    self._bounded_residual[env_ids] = 0.0
    self._phase[env_ids] = self.cfg.safe_lower
