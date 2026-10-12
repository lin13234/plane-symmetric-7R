"""Joint2 重构任务的杆件碰撞分类。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

CollisionPair = tuple[str, str]

@dataclass(frozen=True, slots=True)
class CollisionSample:
  """某一主相位下检测到的禁止实体杆件碰撞。"""
  phase: float
  forbidden_pairs: frozenset[CollisionPair]

def normalize_pair(body_a: str, body_b: str) -> CollisionPair:
  """将无序 body 对规范化，保证 XML 与审计报告的键一致。"""
  return tuple(sorted((body_a, body_b)))  # type: ignore[return-value]

FORBIDDEN_SELF_COLLISION_PAIRS: frozenset[CollisionPair] = frozenset({normalize_pair("L-link1", "R-link1")})
EXCLUDED_TOPOLOGY_PAIRS: frozenset[CollisionPair] = frozenset({normalize_pair("link0", "R-link1")})

def classify_pair(body_a: str, body_b: str) -> str:
  """返回杆件对的碰撞语义分类。"""
  pair = normalize_pair(body_a, body_b)
  if pair in FORBIDDEN_SELF_COLLISION_PAIRS:
    return "forbidden"
  if pair in EXCLUDED_TOPOLOGY_PAIRS:
    return "topology_excluded"
  return "other"

def safe_phase_intervals(samples: Sequence[CollisionSample]) -> tuple[tuple[float, float], ...]:
  """返回离散采样中连续、且没有禁止碰撞的相位闭区间。"""
  intervals: list[tuple[float, float]] = []
  start: float | None = None
  end: float | None = None
  for sample in samples:
    if sample.forbidden_pairs:
      if start is not None and end is not None:
        intervals.append((start, end))
      start = None
      end = None
    else:
      if start is None:
        start = sample.phase
      end = sample.phase
  if start is not None and end is not None:
    intervals.append((start, end))
  return tuple(intervals)

def scan_reference_collisions(xml_path: str, phases: Sequence[float]) -> list[CollisionSample]:
  """用 MuJoCo 逐相位检查历史 Joint2 参考的禁止实体杆件接触。"""
  import mujoco

  from .reference import joint2_reference

  model = mujoco.MjModel.from_xml_path(xml_path)
  data = mujoco.MjData(model)
  joint_names = ("joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint1")
  qpos_indices = tuple(
    int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
    for name in joint_names
  )
  samples: list[CollisionSample] = []
  for phase in phases:
    for index, target in zip(qpos_indices, joint2_reference(phase), strict=True):
      data.qpos[index] = target
    mujoco.mj_forward(model, data)
    forbidden: set[CollisionPair] = set()
    for contact_index in range(data.ncon):
      contact = data.contact[contact_index]
      body_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1])
      body_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2])
      if body_a is not None and body_b is not None and classify_pair(body_a, body_b) == "forbidden":
        forbidden.add(normalize_pair(body_a, body_b))
    samples.append(CollisionSample(float(phase), frozenset(forbidden)))
  return samples