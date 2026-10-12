"""Stage2 技能奖励：纯加性项，无跨项门控。

每项奖励函数返回原始量（通常 ≤ 0 的损失或 [0,1] 的收益），权重在
config 的 RewardTermCfg.weight 中给出。与旧 class2-1 的乘性门控 + 动作
激励结构完全解耦：
  - 前进收益与跟踪误差不再相乘；
  - 没有 reward_excitation（|action| 正收益通道已删除）；
  - 残差幅度与变化率以二次惩罚存在。
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from .config_types import SkillSpec
from .phase import PhaseSpec, phase_state
from .reference import FourierReference


def make_forward_velocity(spec: SkillSpec):
  """前进收益：沿技能定义的前进轴的投影速度，clamp 到 [0, 1.5]。"""

  def forward_velocity(
    env: ManagerBasedRlEnv,
    robot_cfg,
    command_name: str | None = None,
  ) -> torch.Tensor:
    vel_y = env.scene[robot_cfg.name].data.root_com_lin_vel_w[:, 1]
    if command_name is None:
      direction = torch.ones_like(vel_y)
      active = torch.ones_like(vel_y)
    else:
      command = env.command_manager.get_term(command_name)
      direction = command.direction
      # 落地窗口内不结算前进收益；settled 后按 stop 状态门控。
      active = (~command.stop & command.settled).to(vel_y.dtype)
    projected = spec.forward_sign * direction * vel_y
    # 速度门控（历史 class2_1 的 is_moving）：前进速度低于 0.05 m/s 时无前进收益，
    # 逼机构"蹬地前进"而不是"落地慢蹭"。原版为纯加性，无此门控。
    is_moving = (projected > 0.05).to(vel_y.dtype)
    return active * is_moving * torch.clamp(projected, min=0.0, max=1.5)

  return forward_velocity


def make_speed_tracking(spec: SkillSpec):
  """速度跟踪：沿前进轴的投影速度与目标速度的平方误差（≤0）。"""

  def speed_tracking(
    env: ManagerBasedRlEnv,
    robot_cfg,
    command_name: str | None = None,
  ) -> torch.Tensor:
    vel_y = env.scene[robot_cfg.name].data.root_com_lin_vel_w[:, 1]
    if command_name is None:
      direction = torch.ones_like(vel_y)
      speed_scale = torch.ones_like(vel_y)
      active = torch.ones_like(vel_y)
    else:
      command = env.command_manager.get_term(command_name)
      direction = command.direction
      speed_scale = command.speed_scale
      active = (~command.stop).to(vel_y.dtype)
    v_f = spec.forward_sign * direction * vel_y
    target = float(spec.target_speed) * speed_scale * active
    value = -torch.square(v_f - target)
    if command_name is not None:
      # 落地窗口内速度跟踪奖励为 0，避免把落地速度当成跟踪误差。
      command = env.command_manager.get_term(command_name)
      value = value * command.settled.to(value.dtype)
    return value

  return speed_tracking


def make_lateral_drift():
  """横向漂移诊断/消融项：横向（世界 X）速度平方的负值（≤0）。

  该项默认权重为 0.0（由 config 的 RewardTermCfg.weight 给定），横漂仅
  记录为诊断信号，不作为默认训练惩罚；启用时作为可选正则/消融项。
  """

  def lateral_drift(
    env: ManagerBasedRlEnv,
    robot_cfg,
    command_name: str | None = None,
  ) -> torch.Tensor:
    vel_x = env.scene[robot_cfg.name].data.root_com_lin_vel_w[:, 0]
    value = -torch.square(vel_x)
    if command_name is not None:
      # 落地窗口内的横向速度不属于策略决策，不计入漂移惩罚。
      command = env.command_manager.get_term(command_name)
      value = value * command.settled.to(value.dtype)
    return value

  return lateral_drift


def make_reference_tracking(reference_name: str, spec: PhaseSpec):
  """参考跟踪：七关节 wrapped 位置误差绝对值的均值（≤0）。

  参考目标用显式相位计算（与动作链同源），因此这是对"名义轨迹执行
  质量"的直接度量，而不是对任意停位的奖励。
  """
  reference = FourierReference(reference_name)

  def reference_tracking(
    env: ManagerBasedRlEnv,
    robot_cfg,
    command_name: str | None = None,
  ) -> torch.Tensor:
    qpos = env.scene[robot_cfg.name].data.joint_pos  # (N, 7)
    if command_name is None:
      phase, _ = phase_state(env.episode_length_buf, env.step_dt, spec)
    else:
      command = env.command_manager.get_term(command_name)
      phase, _ = command.phase_state(spec)
    qpos_target, _ = reference.qpos_targets(phase, torch.zeros_like(phase))
    err = torch.remainder(qpos - qpos_target + torch.pi, 2.0 * torch.pi) - torch.pi
    value = -torch.mean(torch.abs(err), dim=1)
    if command_name is not None:
      # 落地窗口内跟踪误差不结算，落地姿态不惩罚策略。
      command = env.command_manager.get_term(command_name)
      value = value * command.settled.to(value.dtype)
    return value

  return reference_tracking


def make_residual_magnitude():
  """残差幅值惩罚：四电机策略残差的 L2 平方（≤0）。"""

  def residual_magnitude(
    env: ManagerBasedRlEnv,
    command_name: str | None = None,
  ) -> torch.Tensor:
    action = env.action_manager.action
    value = -torch.sum(torch.square(action), dim=1)
    if command_name is not None:
      # 落地窗口内的动作不进入残差惩罚（此时残差被动作链忽略）。
      command = env.command_manager.get_term(command_name)
      value = value * command.settled.to(value.dtype)
    return value

  return residual_magnitude


def make_residual_rate():
  """残差变化率惩罚：相邻两步策略残差差值的 L2 平方（≤0）。"""

  def residual_rate(
    env: ManagerBasedRlEnv,
    command_name: str | None = None,
  ) -> torch.Tensor:
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    value = -torch.sum(torch.square(action - prev), dim=1)
    if command_name is not None:
      # 落地窗口内残差变化率不结算。
      command = env.command_manager.get_term(command_name)
      value = value * command.settled.to(value.dtype)
    return value

  return residual_rate
