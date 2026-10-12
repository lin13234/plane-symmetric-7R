"""当前实体样机的可执行分支拓扑和合法技能路径规划。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .command import BranchId, SkillId


class UnsupportedBranchError(ValueError):
  """请求了当前样机不可执行或未知的运动分支。"""


def coerce_prototype_branch(value: BranchId | str) -> BranchId:
  """解析样机分支，并区分实体干涉分支与普通拼写错误。"""
  if isinstance(value, BranchId):
    return value
  if value in {"B_S2", "B_S3", "B_S4"}:
    raise UnsupportedBranchError(
      f"{value} 在当前样机中发生实体杆件干涉，不属于可执行子图"
    )
  try:
    return BranchId(value)
  except ValueError as exc:
    raise UnsupportedBranchError(f"未知样机分支: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class TopologyEdge:
  """一条由单个技能实现的有向构型转移。"""

  source: BranchId
  target: BranchId
  skill_id: SkillId


class ExecutableSkillTopology:
  """只在合法样机分支图上返回技能序列，不生成电机动作。"""

  def __init__(
    self,
    edges: tuple[TopologyEdge, ...],
    local_skills: dict[BranchId, tuple[SkillId, ...]],
  ) -> None:
    self.edges = edges
    self.local_skills = local_skills

  def available_skills(self, branch: BranchId | str) -> tuple[SkillId, ...]:
    """列出当前分支可以直接执行的局部技能和出边技能。"""
    branch_id = coerce_prototype_branch(branch)
    edge_skills = tuple(
      edge.skill_id for edge in self.edges if edge.source is branch_id
    )
    return self.local_skills.get(branch_id, ()) + edge_skills

  def plan(
    self,
    source: BranchId | str,
    target: BranchId | str,
  ) -> tuple[SkillId, ...]:
    """通过广度优先搜索返回最短合法技能序列。"""
    source_id = coerce_prototype_branch(source)
    target_id = coerce_prototype_branch(target)
    if source_id is target_id:
      return ()

    queue = deque([(source_id, ())])
    visited = {source_id}
    while queue:
      current, skills = queue.popleft()
      for edge in self.edges:
        if edge.source is not current or edge.target in visited:
          continue
        next_skills = skills + (edge.skill_id,)
        if edge.target is target_id:
          return next_skills
        visited.add(edge.target)
        queue.append((edge.target, next_skills))
    raise UnsupportedBranchError(
      f"当前样机不存在 {source_id.value} 到 {target_id.value} 的合法技能路径"
    )


EXECUTABLE_TOPOLOGY = ExecutableSkillTopology(
  edges=(
    TopologyEdge(BranchId.BP, BranchId.G1, SkillId.APPROACH_GATEWAY_G1),
    TopologyEdge(BranchId.G1, BranchId.BS1, SkillId.SWITCH_BP_TO_BS1),
    TopologyEdge(BranchId.BS1, BranchId.G1, SkillId.APPROACH_GATEWAY_G1),
    TopologyEdge(BranchId.G1, BranchId.BP, SkillId.SWITCH_BS1_TO_BP),
  ),
  local_skills={
    BranchId.BP: (
      SkillId.BP_RECIPROCATING_CRAWL,
      SkillId.HOLD_BRANCH_CONFIGURATION,
    ),
    BranchId.G1: (SkillId.HOLD_BRANCH_CONFIGURATION,),
    BranchId.BS1: (
      SkillId.BS1_RECIPROCATING_CRAWL,
      SkillId.BS1_ROTARY_CRAWL,
      SkillId.HOLD_BRANCH_CONFIGURATION,
    ),
  },
)
