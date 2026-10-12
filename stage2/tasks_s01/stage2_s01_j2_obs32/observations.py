"""S01真实状态/参考观测；输入布局与其它Stage2的32维协议一致。

方向/速度倍率/停止是明确的固定运行域，不代表已经训练过调速和倒车。
J2相位是名义关节参数而非实测角；所有参考使用原动作相同的episode时钟。
"""
import torch
from mjlab.tasks.stage2_joint2_rebuilt.reference import joint2_reference_torch

ACTION_NAME = 'safe_joint2_residual'
TARGET_SPEED = .05  # 原Stage2 SkillSpec默认的名义轴速度；本版不增加速度跟踪奖励。


def phase_state(env):
    """返回下一次动作使用的J2参考相位及三角波单侧导数(rad/s)。

    换向点采用原动作的cycle<=0.5约定；拐点处导数数学上不连续，
    本函数返回对应离散控制侧的值，不把它解释为实测关节速度。
    """
    cfg = env.action_manager.get_term(ACTION_NAME).cfg
    time_s = env.episode_length_buf.to(torch.float32) * env.step_dt
    cycle = torch.remainder(time_s, cfg.period_s) / cfg.period_s
    triangle = torch.where(cycle <= .5, 2. * cycle, 2. * (1. - cycle))
    phase = cfg.safe_lower + triangle * (cfg.safe_upper - cfg.safe_lower)
    speed = 2. * (cfg.safe_upper - cfg.safe_lower) / cfg.period_s
    velocity = torch.where(cycle <= .5, torch.full_like(phase, speed), torch.full_like(phase, -speed))
    return phase, velocity


def root_velocity(env, robot_cfg):
    """根部链接的机体系线速度xyz及Z角速度；不混用世界系或质心速度。"""
    data = env.scene[robot_cfg.name].data
    return torch.cat((data.root_link_lin_vel_b, data.root_link_ang_vel_b[:, 2:3]), dim=1)


def tracking_error(env, robot_cfg):
    """七关节相对于正确J2拟合参考的wrapped位置误差，按XML顺序。"""
    phase, _ = phase_state(env)
    q = env.scene[robot_cfg.name].data.joint_pos
    target = joint2_reference_torch(phase)
    return torch.remainder(q - target + torch.pi, 2. * torch.pi) - torch.pi


def phase_command(env, robot_cfg):
    """真实名义相位/导数及显式固定命令：[sin,cos,导数,目标速度,+1,1,0]。"""
    del robot_cfg
    phase, vel = phase_state(env)
    one = torch.ones_like(phase)
    return torch.stack((phase.sin(), phase.cos(), vel, one * TARGET_SPEED, one, one, torch.zeros_like(phase)), dim=1)
