"""独立的 Stage2-J2 平面残差技能实现。"""
from mjlab.tasks.registry import register_mjlab_task
from .config import TASK_ID, make_joint2_env_cfg, make_joint2_ppo_cfg

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=make_joint2_env_cfg(play=False),
  play_env_cfg=make_joint2_env_cfg(play=True),
  rl_cfg=make_joint2_ppo_cfg(),
)