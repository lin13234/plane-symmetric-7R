"""Stage2 技能观测：显式相位命令、参考跟踪误差与根节点速度。

所有观测函数签名统一为 ``(env, robot_cfg)``，与 ObservationTermCfg 的
``params={"robot_cfg": ...}`` 调用约定一致。关键设计：相位命令由 episode
时钟显式生成（phase_state），不读取任何关节位置，因此策略无法通过停住
驱动关节来冻结参考轨迹。
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from .phase import PhaseSpec, phase_state
from .reference import FourierReference


def make_phase_command_obs(spec: PhaseSpec, target_speed: float):
  """返回相位和运行命令观测函数（7 维）。"""

  def phase_command(
    env: ManagerBasedRlEnv,
    robot_cfg,
    command_name: str | None = None,
  ) -> torch.Tensor:
    del robot_cfg
    if command_name is None:
      phase, phase_vel = phase_state(env.episode_length_buf, env.step_dt, spec)
      direction = torch.ones_like(phase)
      speed_scale = torch.ones_like(phase)
      stop = torch.zeros_like(phase)
      settled = torch.ones_like(phase)
    else:
      command = env.command_manager.get_term(command_name)
      phase, phase_vel = command.phase_state(spec)
      direction = command.direction
      speed_scale = command.speed_scale
      stop = command.stop.to(phase.dtype)
      settled = command.settled.to(phase.dtype)
    # 落地窗口内期望速度置零，策略不会误以为需要立即跟踪名义速度。
    desired_velocity = (
      float(target_speed) * direction * speed_scale * (1.0 - stop) * settled
    )
    return torch.stack(
      [
        torch.sin(phase),
        torch.cos(phase),
        phase_vel,
        desired_velocity,
        direction,
        speed_scale,
        stop,
      ],
      dim=1,
    )

  return phase_command


def make_tracking_error_obs(reference_name: str, spec: PhaseSpec):
  """返回观测函数：七关节相对显式相位参考的 wrapped 位置误差（7 维）。

  参考目标用与动作链完全相同的 phase_state 计算，保证观测与执行一致。
  """
  reference = FourierReference(reference_name)

  def tracking_error(
    env: ManagerBasedRlEnv,
    robot_cfg,
    command_name: str | None = None,
  ) -> torch.Tensor:
    qpos = env.scene[robot_cfg.name].data.joint_pos  # XML 顺序 (N, 7)
    if command_name is None:
      phase, _ = phase_state(env.episode_length_buf, env.step_dt, spec)
    else:
      command = env.command_manager.get_term(command_name)
      phase, _ = command.phase_state(spec)
    qpos_target, _ = reference.qpos_targets(
      phase, torch.zeros_like(phase)
    )  # XML 顺序 (N, 7)
    return torch.remainder(qpos - qpos_target + torch.pi, 2.0 * torch.pi) - torch.pi

  return tracking_error


def make_root_velocity_obs():
  """返回观测函数：机体系线速度(3) + 机体系偏航角速度(1) = 4 维。"""

  def root_velocity(env: ManagerBasedRlEnv, robot_cfg) -> torch.Tensor:
    asset = env.scene[robot_cfg.name]
    lin_vel_b = asset.data.root_link_lin_vel_b  # (N, 3)
    ang_vel_b = asset.data.root_link_ang_vel_b  # (N, 3)
    return torch.cat([lin_vel_b, ang_vel_b[:, 2:3]], dim=1)

  return root_velocity
