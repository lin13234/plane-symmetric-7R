from pathlib import Path
"""独立 Stage2-J2 平面残差技能的环境配置。"""

from __future__ import annotations

import math

import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, time_out
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from .actions import Joint2SafeResidualActionCfg

TASK_ID = "stage2-joint2-planar-residual-v2"
XML_PATH = (
  str(Path(__file__).resolve().parents[4] / "model") + "/"
  "stage2_joint2_rebuilt/zhengti_joint2_v2.xml"
)


def _forward_velocity(env, robot_cfg: SceneEntityCfg) -> torch.Tensor:
  """历史平面基线沿 -Y 的根部实际速度；斜向速度仍可产生正投影。"""
  return -env.scene[robot_cfg.name].data.root_com_lin_vel_w[:, 1]


def _lateral_drift(env, robot_cfg: SceneEntityCfg) -> torch.Tensor:
  """横向速度仅记录；默认权重为零。"""
  return torch.abs(env.scene[robot_cfg.name].data.root_com_lin_vel_w[:, 0])


def _distance_safety(env, robot_cfg: SceneEntityCfg) -> torch.Tensor:
  """仅阻止数值飞离；正常回合由 time_out 截断。"""
  root_xy = env.scene[robot_cfg.name].data.root_com_pos_w[:, :2]
  origin_xy = env.scene.env_origins[:, :2]
  return torch.linalg.vector_norm(root_xy - origin_xy, dim=1) >= 5.0


def make_joint2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """构造隔离的安全 Joint2 平面残差环境。"""
  import mujoco

  robot_cfg = SceneEntityCfg(TASK_ID)
  obs = {
    "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": robot_cfg}),
    "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": robot_cfg}),
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=None,
      entities={
        TASK_ID: EntityCfg(
          spec_fn=lambda: mujoco.MjSpec.from_file(XML_PATH),
          articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=("joint.*",)),)
          ),
        )
      },
      num_envs=1 if play else 2048,
      env_spacing=2.0,
    ),
    observations={
      "actor": ObservationGroupCfg(terms=obs),
      "critic": ObservationGroupCfg(terms=obs),
    },
    actions={"safe_joint2_residual": Joint2SafeResidualActionCfg(entity_name=TASK_ID)},
    rewards={
      "forward_velocity": RewardTermCfg(func=_forward_velocity, weight=20.0, params={"robot_cfg": robot_cfg}),
      "lateral_drift": RewardTermCfg(func=_lateral_drift, weight=0.0, params={"robot_cfg": robot_cfg}),
    },
    terminations={
      "time_out": TerminationTermCfg(func=time_out, time_out=True),
      "distance_safety": TerminationTermCfg(func=_distance_safety, params={"robot_cfg": robot_cfg}),
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.0005, solver="newton", iterations=100, tolerance=1.0e-8), nconmax=2000, njmax=2000),
    viewer=ViewerConfig(entity_name=TASK_ID, distance=2.0) if play else None,
    decimation=16,
    episode_length_s=26.0,
  )


def make_joint2_ppo_cfg(seed: int = 42):
  """四电机小残差控制的保守 PPO 配置。"""
  from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

  return RslRlOnPolicyRunnerCfg(
    seed=seed,
    actor=RslRlModelCfg(hidden_dims=(256, 128), activation="elu", distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(256, 128), activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4, learning_rate=3.0e-4, gamma=0.99, lam=0.95, max_grad_norm=1.0),
    num_steps_per_env=512,
    max_iterations=800,
    save_interval=30,
    experiment_name="Stage2_Joint2_Planar_Residual_V2",
  )