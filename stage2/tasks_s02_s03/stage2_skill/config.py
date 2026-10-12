"""Stage2 技能层共享配置：三个技能规格 + 环境/PPO 工厂。

本模块是三个 Stage2 任务唯一的配置来源：
  - SKILL_SPECS 冻结技能定义（参考、相位、方向、速度）；
  - make_stage2_env_cfg 构造"名义 Fourier + 有界残差"动作链与加性奖励；
  - make_stage2_ppo_cfg 提供适合残差控制的保守探索参数。
"""

from __future__ import annotations

import math

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, time_out
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from .actions import FourierResidualJointPositionActionCfg
from .command import SkillId, Stage2LocomotionCommandCfg
from .config_types import SkillSpec
from .observations import (
  make_phase_command_obs,
  make_root_velocity_obs,
  make_tracking_error_obs,
)
from .phase import PhaseSpec
from .reference import ACTUATED_JOINT_NAMES
from .rewards import (
  make_forward_velocity,
  make_lateral_drift,
  make_reference_tracking,
  make_residual_magnitude,
  make_residual_rate,
  make_speed_tracking,
)

# 三个冻结技能（相位速度为目标名义值，冒烟训练后按真实周期调整）。
# forward_sign 由 2026-09-17 零残差基线实测冻结：B_S1 系列（rotary/inchworm）
# 名义驱动实际向 +Y 移动；B_P（planar-inchworm）向 -Y 移动。
SKILL_SPECS: dict[str, SkillSpec] = {
  "stage2-spatial-rotary": SkillSpec(
    task_id="stage2-spatial-rotary",
    skill_id=SkillId.BS1_ROTARY_CRAWL,
    reference_name="spatial_reference_B_S1",
    phase=PhaseSpec(mode="rotary", speed=0.8),
    episode_length_s=25.0,
    forward_sign=1.0,
  ),
  "stage2-spatial-inchworm": SkillSpec(
    task_id="stage2-spatial-inchworm",
    skill_id=SkillId.BS1_RECIPROCATING_CRAWL,
    reference_name="spatial_reference_B_S1",
    phase=PhaseSpec(mode="inchworm", speed=1.0, limit=math.pi / 2),
    episode_length_s=25.0,
    forward_sign=1.0,
  ),
  "stage2-planar-inchworm": SkillSpec(
    task_id="stage2-planar-inchworm",
    skill_id=SkillId.BP_RECIPROCATING_CRAWL,
    reference_name="planar_reference_B_P",
    phase=PhaseSpec(mode="inchworm", speed=1.0, limit=math.pi),
    episode_length_s=25.0,
    forward_sign=-1.0,
  ),
}


def make_distance_termination(target_distance: float = 5.0):
  """返回安全 termination：相对本环境起点的水平位移达到 target_distance 即终止。

  5 m 仅用于拦截数值异常或飞离起点的轨迹；正常回合由 time_out 结束。
  """

  def dist_term(env, robot_cfg):
    import torch

    root_xy = env.scene[robot_cfg.name].data.root_com_pos_w[:, :2]
    origin_xy = env.scene.env_origins[:, :2]
    return torch.linalg.vector_norm(root_xy - origin_xy, dim=1) >= target_distance

  return dist_term

def make_stage2_env_cfg(spec: SkillSpec, play: bool = False) -> ManagerBasedRlEnvCfg:
  """构造一个 Stage2 技能环境。

  - 动作链：Fourier 名义四电机目标 + clamp 到 [-1,1] 的残差 × residual_scale；
  - 观测：关节位置/速度、根速度、参考跟踪误差、显式相位命令；
  - 奖励：六个独立加性项（前进/速度跟踪/横向漂移/参考跟踪/残差幅值/残差变化率）；
    横向漂移项默认权重 0.0，仅作诊断/消融，不作为默认训练惩罚。
  - 终止：26 s 正常 time_out + 相对起点 5 m 的异常飞走保护。
  """
  command_name = "skill"
  command_cfg = Stage2LocomotionCommandCfg(
    # 8 s episode 内不自动改命令；每次环境 reset 时重新采样。
    resampling_time_range=(1000.0, 1000.0),
    skill_id=spec.skill_id,
    direction_choices=(1,) if play else (-1, 1),
    speed_scale_range=(1.0, 1.0) if play else (0.5, 1.5),
    # reset 后先经历固定落地窗口，窗口内相位冻结、残差与奖励门控为 0。
    settling_time_s=spec.settling_time_s,
  )

  robot_cfg = SceneEntityCfg(spec.task_id)

  def load_spec():
    import mujoco

    return mujoco.MjSpec.from_file(spec.xml_path)

  obs_terms = {
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel, params={"asset_cfg": robot_cfg}
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel, params={"asset_cfg": robot_cfg}
    ),
    "root_velocity": ObservationTermCfg(
      func=make_root_velocity_obs(), params={"robot_cfg": robot_cfg}
    ),
    "tracking_error": ObservationTermCfg(
      func=make_tracking_error_obs(spec.reference_name, spec.phase),
      params={"robot_cfg": robot_cfg, "command_name": command_name},
    ),
    "phase_command": ObservationTermCfg(
      func=make_phase_command_obs(spec.phase, spec.target_speed),
      params={"robot_cfg": robot_cfg, "command_name": command_name},
    ),
  }

  rewards = {
    "forward_velocity": RewardTermCfg(
      func=make_forward_velocity(spec),
      weight=20.0,
      params={"robot_cfg": robot_cfg, "command_name": command_name},
    ),
    "speed_tracking": RewardTermCfg(
      func=make_speed_tracking(spec),
      weight=10.0,
      params={"robot_cfg": robot_cfg, "command_name": command_name},
    ),
    # 横向漂移项保留但默认权重 0.0：横漂仅记录作诊断/消融，不惩罚斜向推进。
    "lateral_drift": RewardTermCfg(
      func=make_lateral_drift(),
      weight=spec.lateral_drift_weight,
      params={"robot_cfg": robot_cfg, "command_name": command_name},
    ),
    "reference_tracking": RewardTermCfg(
      func=make_reference_tracking(spec.reference_name, spec.phase),
      weight=1.0,
      params={"robot_cfg": robot_cfg, "command_name": command_name},
    ),
    "residual_magnitude": RewardTermCfg(
      func=make_residual_magnitude(),
      weight=0.1,
      params={"command_name": command_name},
    ),
    "residual_rate": RewardTermCfg(
      func=make_residual_rate(),
      weight=0.5,
      params={"command_name": command_name},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      # 不再额外添加 Scene 地面：XML 自带地面及其摩擦/接触参数是唯一接触面。
      terrain=None,
      entities={
        spec.task_id: EntityCfg(
          spec_fn=load_spec,
          # 恢复 XML 悬空初态（root 高度 0.40 m），落地过程由物理自然完成。
          init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.40)),
          articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=("joint.*",)),)
          ),
        )
      },
      num_envs=1 if play else 2048,
      env_spacing=2,
    ),
    observations={
      "actor": ObservationGroupCfg(terms=obs_terms),
      "critic": ObservationGroupCfg(terms=obs_terms),
    },
    commands={command_name: command_cfg},
    actions={
      "fourier_residual": FourierResidualJointPositionActionCfg(
        entity_name=spec.task_id,
        actuator_names=ACTUATED_JOINT_NAMES,
        reference_name=spec.reference_name,
        phase_spec=spec.phase,
        command_name=command_name,
        residual_scale=0.05,
      )
    },
    rewards=rewards,
    terminations={
      # 正常 26 s 采样边界：truncation，不作为失败终止。
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
      # 仅保护异常飞走；不会替代正常的 time_out。
      "distance_safety": TerminationTermCfg(
        func=make_distance_termination(5.0),
        params={"robot_cfg": robot_cfg},
      ),
    },
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.0005, solver="newton", iterations=100, tolerance=1e-8
      ),
      nconmax=2000,
      njmax=2000,
    ),
    viewer=(ViewerConfig(entity_name=spec.task_id, distance=2.0) if play else None),
    decimation=16,
    # 落地窗口额外占用 settling_time_s，这里相应延长，保证有效技能时长不缩短。
    episode_length_s=spec.episode_length_s + spec.settling_time_s,
  )


def make_stage2_ppo_cfg(experiment_name: str, seed: int = 42):
  """残差控制专用的保守探索 PPO 配置。

  与旧 class2-1（init_std=1.2, entropy=0.06, lr=3e-4, steps=512）不同：
  残差被 clamp 在 ±1 且乘 residual_scale，因此需要更小的初始标准差和
  更低的熵，避免策略一开始就利用名义轨迹加满幅残差。
  """
  from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
  )

  return RslRlOnPolicyRunnerCfg(
    seed=seed,
    actor=RslRlModelCfg(
      hidden_dims=(256, 128),
      activation="elu",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(hidden_dims=(256, 128), activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      gamma=0.99,
      lam=0.95,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=128,
    max_iterations=800,
  )
