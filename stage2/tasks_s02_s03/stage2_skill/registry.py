"""Stage2 技能名称、实现方式和验证状态的唯一注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .command import BranchId, SkillId


class ImplementationKind(str, Enum):
  LEARNED = "learned"
  DETERMINISTIC = "deterministic"


class SkillStatus(str, Enum):
  IMPLEMENTED = "implemented"
  EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class SkillDefinition:
  """上层规划器和实验脚本共享的技能元数据。"""

  skill_id: SkillId
  code_name: str
  chinese_name: str
  source_branch: BranchId
  target_branch: BranchId
  implementation: ImplementationKind
  status: SkillStatus
  task_id: str | None = None
  prerequisites: tuple[str, ...] = ()


SKILL_REGISTRY: dict[SkillId, SkillDefinition] = {
  SkillId.BP_RECIPROCATING_CRAWL: SkillDefinition(
    skill_id=SkillId.BP_RECIPROCATING_CRAWL,
    code_name="bp_reciprocating_crawl",
    chinese_name="平面分支往复爬行",
    source_branch=BranchId.BP,
    target_branch=BranchId.BP,
    implementation=ImplementationKind.LEARNED,
    status=SkillStatus.IMPLEMENTED,
    task_id="stage2-planar-inchworm",
  ),
  SkillId.BS1_RECIPROCATING_CRAWL: SkillDefinition(
    skill_id=SkillId.BS1_RECIPROCATING_CRAWL,
    code_name="bs1_reciprocating_crawl",
    chinese_name="空间分支往复爬行",
    source_branch=BranchId.BS1,
    target_branch=BranchId.BS1,
    implementation=ImplementationKind.LEARNED,
    status=SkillStatus.IMPLEMENTED,
    task_id="stage2-spatial-inchworm",
  ),
  SkillId.BS1_ROTARY_CRAWL: SkillDefinition(
    skill_id=SkillId.BS1_ROTARY_CRAWL,
    code_name="bs1_rotary_crawl",
    chinese_name="空间分支连续旋转爬行",
    source_branch=BranchId.BS1,
    target_branch=BranchId.BS1,
    implementation=ImplementationKind.LEARNED,
    status=SkillStatus.IMPLEMENTED,
    task_id="stage2-spatial-rotary",
  ),
  SkillId.APPROACH_GATEWAY_G1: SkillDefinition(
    skill_id=SkillId.APPROACH_GATEWAY_G1,
    code_name="approach_gateway_g1",
    chinese_name="G1过渡构型接近",
    source_branch=BranchId.BP,
    target_branch=BranchId.G1,
    implementation=ImplementationKind.DETERMINISTIC,
    status=SkillStatus.EXPERIMENTAL,
    prerequisites=("validated_g1_nominal_path",),
  ),
  SkillId.SWITCH_BP_TO_BS1: SkillDefinition(
    skill_id=SkillId.SWITCH_BP_TO_BS1,
    code_name="switch_bp_to_bs1",
    chinese_name="平面到空间分支切换",
    source_branch=BranchId.G1,
    target_branch=BranchId.BS1,
    implementation=ImplementationKind.LEARNED,
    status=SkillStatus.EXPERIMENTAL,
    prerequisites=(
      "validated_g1_nominal_path",
      "collision_path_verified",
      "branch_classifier_verified",
    ),
  ),
  SkillId.SWITCH_BS1_TO_BP: SkillDefinition(
    skill_id=SkillId.SWITCH_BS1_TO_BP,
    code_name="switch_bs1_to_bp",
    chinese_name="空间到平面分支切换",
    source_branch=BranchId.G1,
    target_branch=BranchId.BP,
    implementation=ImplementationKind.LEARNED,
    status=SkillStatus.EXPERIMENTAL,
    prerequisites=(
      "validated_g1_nominal_path",
      "collision_path_verified",
      "branch_classifier_verified",
    ),
  ),
  SkillId.HOLD_BRANCH_CONFIGURATION: SkillDefinition(
    skill_id=SkillId.HOLD_BRANCH_CONFIGURATION,
    code_name="hold_branch_configuration",
    chinese_name="目标分支构型保持",
    source_branch=BranchId.G1,
    target_branch=BranchId.G1,
    implementation=ImplementationKind.DETERMINISTIC,
    status=SkillStatus.EXPERIMENTAL,
    prerequisites=("validated_hold_targets",),
  ),
}


_LEARNING_ORDER = (
  SkillId.BP_RECIPROCATING_CRAWL,
  SkillId.BS1_RECIPROCATING_CRAWL,
  SkillId.BS1_ROTARY_CRAWL,
  SkillId.SWITCH_BP_TO_BS1,
  SkillId.SWITCH_BS1_TO_BP,
)


def learned_skill_ids() -> tuple[SkillId, ...]:
  """按冻结训练顺序返回五个学习任务。"""
  return _LEARNING_ORDER
