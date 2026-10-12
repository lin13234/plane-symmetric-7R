"""冻结Fourier参考轨迹及论文坐标到XML坐标的注册。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "fourier_references"
REFERENCE_MANIFEST = REFERENCE_DIR / "reference_manifest.json"

# paper J1..J7 -> XML qpos [joint2, joint3, joint4, joint5, joint6, joint7, joint1]
PAPER_TO_QPOS_SLOT = (6, 0, 1, 2, 3, 4, 5)
H1_SIGN = (1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0)

# Entity.find_joints_by_actuator_names 会按XML自然顺序返回驱动关节，而不是按查询列表顺序。
ACTUATED_JOINT_NAMES = ("joint2", "joint3", "joint7", "joint1")
ACTUATED_QPOS_SLOTS = (0, 1, 5, 6)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def paper_to_qpos(paper_q: torch.Tensor) -> torch.Tensor:
  """把末维为J1..J7的tensor转换为XML七关节顺序。"""
  if paper_q.shape[-1] != 7:
    raise ValueError("paper_q must have seven joints on its last dimension")
  qpos = torch.empty_like(paper_q)
  for paper_index, qpos_slot in enumerate(PAPER_TO_QPOS_SLOT):
    qpos[..., qpos_slot] = H1_SIGN[qpos_slot] * paper_q[..., paper_index]
  return qpos


def select_actuated_targets(qpos_target: torch.Tensor) -> torch.Tensor:
  """按mjlab实际动作通道顺序抽取四个电机的名义位置目标。"""
  if qpos_target.shape[-1] != 7:
    raise ValueError("qpos_target must contain all seven joints")
  return qpos_target[..., list(ACTUATED_QPOS_SLOTS)]


class FourierReference:
  """加载并校验冻结manifest中的一条七关节Fourier参考。"""

  def __init__(self, name: str):
    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    references = manifest["references"]
    if name not in references:
      raise ValueError(f"{name!r} is not a frozen Stage2 reference")

    entry = references[name]
    path = REFERENCE_DIR / f"{name}.json"
    actual_hash = _sha256(path)
    expected_hash = entry["file_sha256"]
    if actual_hash != expected_hash:
      raise ValueError(
        f"reference hash mismatch for {name}: {actual_hash} != {expected_hash}"
      )

    self.name = name
    self.branch = entry["branch"]
    self.path = path
    self.ref = json.loads(path.read_text(encoding="utf-8"))
    self.n_harmonics = int(self.ref["n_harmonics"])
    self._cache: dict[str, dict[str, torch.Tensor]] = {}

  def _coefficients(
    self,
    device: torch.device,
    dtype: torch.dtype,
  ) -> dict[str, torch.Tensor]:
    key = f"{device}:{dtype}"
    if key not in self._cache:
      a0: list[float] = []
      linear: list[float] = []
      cosine: list[list[float]] = []
      sine: list[list[float]] = []
      for joint_index in range(1, 8):
        coeff = self.ref["joints"][f"J{joint_index}"]["coefficients"]
        a0.append(coeff["a0"])
        linear.append(coeff.get("c", 0.0))
        cosine.append(coeff["a"])
        sine.append(coeff["b"])
      self._cache[key] = {
        "a0": torch.tensor(a0, device=device, dtype=dtype),
        "linear": torch.tensor(linear, device=device, dtype=dtype),
        "cosine": torch.tensor(cosine, device=device, dtype=dtype),
        "sine": torch.tensor(sine, device=device, dtype=dtype),
      }
    return self._cache[key]

  def paper_targets(
    self,
    phase: torch.Tensor,
    phase_velocity: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """计算论文J1..J7坐标下的位置与速度目标。"""
    if phase.ndim != 1 or phase_velocity.shape != phase.shape:
      raise ValueError("phase and phase_velocity must be equal-length vectors")

    coeff = self._coefficients(phase.device, phase.dtype)
    harmonics = torch.arange(
      1,
      self.n_harmonics + 1,
      device=phase.device,
      dtype=phase.dtype,
    )
    angles = phase[:, None] * harmonics[None, :]
    cosine = torch.cos(angles)
    sine = torch.sin(angles)

    position = coeff["a0"][None, :] + coeff["linear"][None, :] * phase[:, None]
    position = position + cosine @ coeff["cosine"].T + sine @ coeff["sine"].T

    derivative = coeff["linear"][None, :]
    derivative = derivative - sine @ (coeff["cosine"] * harmonics[None, :]).T
    derivative = derivative + cosine @ (coeff["sine"] * harmonics[None, :]).T
    velocity = derivative * phase_velocity[:, None]
    return position, velocity

  def qpos_targets(
    self,
    phase: torch.Tensor,
    phase_velocity: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """计算XML七关节顺序下的位置与速度目标。"""
    paper_position, paper_velocity = self.paper_targets(phase, phase_velocity)
    return paper_to_qpos(paper_position), paper_to_qpos(paper_velocity)
