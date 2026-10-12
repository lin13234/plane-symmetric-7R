"""把显式Fourier名义轨迹与四电机强化学习残差合成为位置目标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

from .control import compose_residual_target
from .phase import PhaseSpec, phase_state
from .reference import ACTUATED_JOINT_NAMES, FourierReference

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class FourierResidualJointPositionActionCfg(BaseActionCfg):
  """Fourier名义位置加有界策略残差的动作配置。"""

  actuator_names: tuple[str, ...] = ACTUATED_JOINT_NAMES
  reference_name: str
  phase_spec: PhaseSpec
  command_name: str | None = None
  residual_scale: float = 0.05
  scale: float = 1.0
  offset: float = 0.0

  def __post_init__(self) -> None:
    self.transmission_type = TransmissionType.JOINT
    if self.residual_scale < 0.0:
      raise ValueError("residual_scale must be non-negative")

  def build(self, env: ManagerBasedRlEnv) -> FourierResidualJointPositionAction:
    return FourierResidualJointPositionAction(self, env)


class FourierResidualJointPositionAction(BaseAction):
  """每个控制步下发名义四电机目标，并叠加受限策略修正。"""

  cfg: FourierResidualJointPositionActionCfg

  def __init__(
    self,
    cfg: FourierResidualJointPositionActionCfg,
    env: ManagerBasedRlEnv,
  ) -> None:
    super().__init__(cfg=cfg, env=env)
    if tuple(self.target_names) != ACTUATED_JOINT_NAMES:
      raise ValueError(
        "actuated joint order mismatch: "
        f"expected {ACTUATED_JOINT_NAMES}, got {tuple(self.target_names)}"
      )

    self.reference = FourierReference(cfg.reference_name)
    self._bounded_residual = torch.zeros_like(self._raw_actions)
    self._nominal_target = torch.zeros_like(self._raw_actions)
    self._target = torch.zeros_like(self._raw_actions)
    self._phase = torch.zeros(self.num_envs, device=self.device)
    self._phase_velocity = torch.zeros_like(self._phase)

  @property
  def bounded_residual(self) -> torch.Tensor:
    """经[-1,1]限幅、尚未乘物理尺度的策略残差。"""
    return self._bounded_residual

  @property
  def nominal_target(self) -> torch.Tensor:
    """最近一次应用的四电机Fourier名义位置目标。"""
    return self._nominal_target

  @property
  def phase(self) -> torch.Tensor:
    """最近一次应用动作时的显式相位。"""
    return self._phase

  @property
  def phase_velocity(self) -> torch.Tensor:
    """最近一次应用动作时的显式相位速度。"""
    return self._phase_velocity

  def process_actions(self, actions: torch.Tensor) -> None:
    """保存原始策略动作，并在动作项内部强制限幅。"""
    self._raw_actions[:] = actions.to(self.device)
    self._bounded_residual[:] = torch.clamp(self._raw_actions, -1.0, 1.0)
    self._processed_actions[:] = self._bounded_residual

  def apply_actions(self) -> None:
    """按当前episode时钟计算并下发四电机位置目标。"""
    if self.cfg.command_name is None:
      # 保留给旧测试和兼容配置的固定正向相位路径。
      phase, phase_velocity = phase_state(
        self._env.episode_length_buf,
        self._env.step_dt,
        self.cfg.phase_spec,
      )
      settled = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
    else:
      command = self._env.command_manager.get_term(self.cfg.command_name)
      phase, phase_velocity = command.phase_state(self.cfg.phase_spec)
      settled = command.settled
    self._phase[:] = phase
    self._phase_velocity[:] = phase_velocity

    qpos_target, _ = self.reference.qpos_targets(phase, phase_velocity)
    self._nominal_target[:] = qpos_target[:, self.target_ids]
    # 落地中的环境忽略策略残差，只下发 phase=0 名义目标；已 settled 才叠加残差。
    gated_residual = self._bounded_residual * settled.to(
      self._bounded_residual.dtype
    ).unsqueeze(1)
    self._target[:] = compose_residual_target(
      self._nominal_target,
      gated_residual,
      self.cfg.residual_scale,
    )
    self._entity.set_joint_position_target(self._target, joint_ids=self.target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """清空对应环境的动作缓存；相位由episode计数自然归零。"""
    super().reset(env_ids)
    self._bounded_residual[env_ids] = 0.0
    self._nominal_target[env_ids] = 0.0
    self._target[env_ids] = 0.0
    self._phase[env_ids] = 0.0
    self._phase_velocity[env_ids] = 0.0
