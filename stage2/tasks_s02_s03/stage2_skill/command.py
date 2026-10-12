"""Stage2 样机技能的稳定标识和上层运行命令。

本模块只描述当前实体样机可执行的 ``B_P <-> G1 <-> B_S1`` 子图。
理论分支 B_S2/B_S3/B_S4 因当前样机实体干涉，不在枚举中出现。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .phase import PhaseSpec, phase_from_elapsed

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class BranchId(str, Enum):
  """当前样机允许进入的构型分支或公共过渡构型。"""

  BP = "B_P"
  G1 = "G1"
  BS1 = "B_S1"


class SkillId(str, Enum):
  """稳定技能编号；编号进入检查点元数据后不得重新解释。"""

  BP_RECIPROCATING_CRAWL = "S01"
  BS1_RECIPROCATING_CRAWL = "S02"
  BS1_ROTARY_CRAWL = "S03"
  APPROACH_GATEWAY_G1 = "S10"
  SWITCH_BP_TO_BS1 = "S11"
  SWITCH_BS1_TO_BP = "S12"
  HOLD_BRANCH_CONFIGURATION = "S13"


@dataclass(frozen=True, slots=True)
class SkillCommand:
  """大模型或任务规划器传给低层技能的受限命令。

  Args:
      skill_id: 当前要执行的稳定技能编号。
      direction: 技能名义方向的符号，只允许 ``-1`` 或 ``+1``。
      speed_scale: 相对于名义相位速度的倍率，冻结在 [0.5, 1.5]。
      stop: 为真时停止相位推进。停止不等于重置机器人状态。
  """

  skill_id: SkillId
  direction: int = 1
  speed_scale: float = 1.0
  stop: bool = False

  def __post_init__(self) -> None:
    # 接受来自 JSON 的字符串，同时在对象内部统一保存为枚举。
    try:
      skill_id = SkillId(self.skill_id)
    except ValueError as exc:
      raise ValueError(f"unknown prototype skill_id: {self.skill_id!r}") from exc
    object.__setattr__(self, "skill_id", skill_id)

    if self.direction not in (-1, 1):
      raise ValueError("direction must be either -1 or +1")
    if not 0.5 <= float(self.speed_scale) <= 1.5:
      raise ValueError("speed_scale must be within [0.5, 1.5]")
    object.__setattr__(self, "speed_scale", float(self.speed_scale))
    object.__setattr__(self, "stop", bool(self.stop))

  @property
  def signed_speed_scale(self) -> float:
    """返回带方向的速度倍率；停止命令对应零倍率。"""
    if self.stop:
      return 0.0
    return float(self.direction) * self.speed_scale


class Stage2LocomotionCommand(CommandTerm):
  """为每个并行环境维护方向、速度、停止状态和独立技能时钟。"""

  cfg: "Stage2LocomotionCommandCfg"

  def __init__(
    self,
    cfg: "Stage2LocomotionCommandCfg",
    env: "ManagerBasedRlEnv",
  ) -> None:
    super().__init__(cfg, env)
    self.direction = torch.ones(self.num_envs, device=self.device)
    self.speed_scale = torch.ones(self.num_envs, device=self.device)
    self.stop = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    # elapsed 只在非 stop 状态增长，因此 stop 会冻结当前相位而非回零。
    self.elapsed = torch.zeros(self.num_envs, device=self.device)
    self._command_dt = 0.0
    # 落地窗口状态：settled=False 表示仍在落地，相位冻结、残差与奖励门控为 0。
    self.settled = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.settling_time_left = torch.full(
      (self.num_envs,), self.cfg.settling_time_s, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    """返回策略可观察的 [direction, speed_scale, stop]。"""
    return torch.stack(
      [self.direction, self.speed_scale, self.stop.to(self.direction.dtype)],
      dim=1,
    )

  def set_command(
    self,
    env_ids: torch.Tensor,
    command: SkillCommand,
    *,
    reset_phase: bool = True,
  ) -> None:
    """由上层规划器为指定环境设置命令，不重置机器人状态。"""
    if command.skill_id is not self.cfg.skill_id:
      raise ValueError(
        f"command skill {command.skill_id.value} does not match "
        f"term skill {self.cfg.skill_id.value}"
      )
    self.direction[env_ids] = float(command.direction)
    self.speed_scale[env_ids] = command.speed_scale
    self.stop[env_ids] = command.stop
    # 显式上层命令优先于自动采样；否则 time_left=0 会在首步覆盖命令。
    self.time_left[env_ids] = self.cfg.resampling_time_range[1]

    if reset_phase:
      self.elapsed[env_ids] = 0.0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    """reset 时重启落地窗口；命令重采样或上层命令不触发落地。"""
    extras = super().reset(env_ids)
    # 新 episode 从悬空初态开始，必须重新经历落地窗口。
    self.settled[env_ids] = False
    self.settling_time_left[env_ids] = self.cfg.settling_time_s
    return extras

  def phase_state(self, spec: PhaseSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """用命令时钟计算逐环境相位；stop 与落地窗口都只把速度置零。

    落地窗口内 elapsed 保持 0，因此相位本身已为零；这里额外把相位速度置零，
    保证观测里的期望速度与相位速度在落地期间都为 0。
    """
    phase, velocity = phase_from_elapsed(
      self.elapsed,
      spec,
      direction=self.direction,
      speed_scale=self.speed_scale,
    )
    velocity = torch.where(self.stop, torch.zeros_like(velocity), velocity)
    velocity = torch.where(self.settled, velocity, torch.zeros_like(velocity))
    return phase, velocity

  def compute(self, dt: float) -> None:
    self._command_dt = float(dt)
    super().compute(dt)

  def _update_metrics(self) -> None:
    # 当前命令本身无累积误差；运动误差由统一 metrics/reward 记录。
    return None

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    choices = torch.tensor(
      self.cfg.direction_choices,
      device=self.device,
      dtype=self.direction.dtype,
    )
    choice_indices = torch.randint(len(choices), (count,), device=self.device)
    self.direction[env_ids] = choices[choice_indices]
    self.speed_scale[env_ids] = torch.empty(count, device=self.device).uniform_(
      *self.cfg.speed_scale_range
    )
    if self.cfg.stop_probability == 0.0:
      self.stop[env_ids] = False
    else:
      self.stop[env_ids] = (
        torch.rand(count, device=self.device) < self.cfg.stop_probability
      )
    self.elapsed[env_ids] = 0.0

  def _update_command(self) -> None:
    # 只有本步开始前已 settled 且未 stop 的环境才推进技能时钟。
    active = (~self.stop & self.settled).to(self.elapsed.dtype)
    self.elapsed += self._command_dt * active

    # 落地窗口倒计时：未 settled 的环境扣除 dt，扣完即置 settled。
    # settled 在本步末尾才置真，因此本步内 elapsed 不推进，
    # settling 结束后的下一控制步才从 phase=0 启动。
    settling = ~self.settled
    self.settling_time_left = self.settling_time_left - self._command_dt * (
      settling.to(self.settling_time_left.dtype)
    )
    self.settled = self.settled | (settling & (self.settling_time_left <= 0.0))


@dataclass(kw_only=True)
class Stage2LocomotionCommandCfg(CommandTermCfg):
  """并行环境的方向/速度命令采样范围与落地窗口配置。"""

  skill_id: SkillId
  direction_choices: tuple[int, ...] = (-1, 1)
  speed_scale_range: tuple[float, float] = (0.5, 1.5)
  stop_probability: float = 0.0
  settling_time_s: float = 1.0

  def __post_init__(self) -> None:
    self.skill_id = SkillId(self.skill_id)
    if not self.direction_choices or any(
      value not in (-1, 1) for value in self.direction_choices
    ):
      raise ValueError("direction_choices must contain only -1 and/or +1")
    low, high = self.speed_scale_range
    if low < 0.5 or high > 1.5 or low > high:
      raise ValueError("speed_scale_range must stay within [0.5, 1.5]")
    if not 0.0 <= self.stop_probability <= 1.0:
      raise ValueError("stop_probability must be within [0, 1]")
    if self.settling_time_s < 0.0:
      raise ValueError("settling_time_s must be non-negative")

  def build(self, env: "ManagerBasedRlEnv") -> Stage2LocomotionCommand:
    return Stage2LocomotionCommand(self, env)
