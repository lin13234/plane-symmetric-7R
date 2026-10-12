"""独立 Stage2-J2 任务导出。"""
from .config import make_joint2_env_cfg

def stage2_joint2_planar_residual_v2_env_cfg(play: bool = False):
  """供任务注册表构造训练或播放环境。"""
  return make_joint2_env_cfg(play=play)