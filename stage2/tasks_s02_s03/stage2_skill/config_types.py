from pathlib import Path
"""Stage2 技能规格类型：供 config / rewards / 评估器共享，避免循环导入。"""

from __future__ import annotations

from dataclasses import dataclass

from .command import SkillId
from .phase import PhaseSpec

# Stage2 专用 mesh 碰撞 XML：从 classtrain2-1 原模型复制而来，原资产不改动；
# classtrain1-2 是 Stage1 无接触版本，焊死在地面无法前进。
XML_PATH = (
  str(Path(__file__).resolve().parents[3]) + "/"
  "src/mjlab/asset_zoo/classtrain2-1/zhengti_stage2_mesh_collision.xml"
)


@dataclass(frozen=True, slots=True)
class SkillSpec:
  """一个 Stage2 低层运动技能的完整定义。

  Attributes:
      task_id: 注册到 mjlab task registry 的任务名。
      reference_name: 冻结 manifest 中的 Fourier 参考名（必须存在于
          reference_manifest.json）。
      phase: 显式相位模式（rotary 连续 / inchworm 往返）。
      xml_path: 物理资产 XML。
      forward_sign: 前进轴符号（世界 -Y 为 -1，+Y 为 +1）。零残差基线
          验证方向后冻结，不得随意修改。
      target_speed: 速度跟踪奖励的目标前进速度 (m/s)。
      lateral_drift_weight: 横向漂移的可选正则/消融权重，默认 0.0（不
          作为训练惩罚或评估失败条件）；横向漂移量仍由 metrics 记录以
          表征不同运动分支。
      settling_time_s: reset 后落地稳定的固定窗口时长 (s)；窗口内相位冻结、
          残差与奖励门控为 0，episode 时长相应延长该值。
  """

  task_id: str
  skill_id: SkillId
  reference_name: str
  phase: PhaseSpec
  episode_length_s: float
  xml_path: str = XML_PATH
  forward_sign: float = -1.0
  target_speed: float = 0.05
  lateral_drift_weight: float = 0.0
  settling_time_s: float = 1.0
