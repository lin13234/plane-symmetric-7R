"""历史 Joint2 驱动闭链参考的独立纯函数实现。"""
from __future__ import annotations
import math

_COEFFS = tuple(tuple(value * math.pi / 180.0 for value in row) for row in (
(25.8923,-57.1567,-7.0495,-28.8730,5.3610,1.4836,-1.0661,0.9405,-0.1334,-0.5543,0.1513,-0.0518),
(-7.8844,2.1503,4.0853,-0.0190,0.8001,-0.8920,1.1344,-0.2199,-0.2839,0.2285,0.0080,-0.2916),
(-14.5197,-0.0563,-7.3410,17.1770,-3.4315,-2.0354,0.8898,-0.5618,0.0331,0.3721,-0.1101,-0.0042),
(15.4390,-4.2861,-8.2623,0.1740,-1.8320,1.5741,-2.2121,0.4419,0.3970,-0.4500,0.0079,0.5178),
(-11.3907,-0.0849,14.3879,11.7163,-1.9525,0.5367,0.1973,-0.3819,0.0866,0.1965,-0.0374,0.0412),
(-7.6870,2.1344,4.2789,0.0649,0.8441,-0.7738,1.2486,-0.2380,-0.2264,0.3223,0.0083,-0.3478),
))

def _features(phase: float) -> tuple[float, ...]:
  values = [1.0, phase]
  for harmonic in range(1, 6):
    values.extend((math.sin(harmonic * phase), math.cos(harmonic * phase)))
  return tuple(values)

def joint2_reference(phase: float) -> tuple[float, ...]:
  """返回 XML 顺序 (joint2, joint3, joint4, joint5, joint6, joint7, joint1)。"""
  features = _features(phase)
  j1, j3, j4, j5, j6, j7 = (sum(c * f for c, f in zip(row, features, strict=True)) for row in _COEFFS)
  return (phase, j3, j4, j5, j6, j7, j1)
from dataclasses import dataclass

# 2026-09-23 的 721 点自碰撞审计给出的唯一连续可行区间；端点向内收一个采样步长。
DEFAULT_SAFE_PHASE_INTERVALS: tuple[tuple[float, float], ...] = ((-1.492257, 2.303835),)

@dataclass(frozen=True, slots=True)
class Joint2FeasibleReference:
  """把请求主相位限制到经实体杆件碰撞审计的安全窗口。"""
  safe_intervals: tuple[tuple[float, float], ...] = DEFAULT_SAFE_PHASE_INTERVALS

  def project_phase(self, requested_phase: float) -> float:
    """返回请求相位本身或最近的安全边界。"""
    for lower, upper in self.safe_intervals:
      if lower <= requested_phase <= upper:
        return requested_phase
    boundaries = tuple(boundary for interval in self.safe_intervals for boundary in interval)
    return min(boundaries, key=lambda boundary: abs(boundary - requested_phase))

  def targets(self, requested_phase: float) -> tuple[float, ...]:
    """返回安全主相位对应的七关节名义参考。"""
    return joint2_reference(self.project_phase(requested_phase))

def safe_inchworm_phase(
  time_s: float,
  reference: Joint2FeasibleReference = Joint2FeasibleReference(),
  *,
  period_s: float = 4.0,
) -> float:
  """在唯一连续安全窗口中生成三角波式往复主相位。"""
  if period_s <= 0.0:
    raise ValueError("period_s must be positive")
  if len(reference.safe_intervals) != 1:
    raise ValueError("safe inchworm reference requires one continuous interval")
  lower, upper = reference.safe_intervals[0]
  cycle = (time_s % period_s) / period_s
  triangle = 2.0 * cycle if cycle <= 0.5 else 2.0 * (1.0 - cycle)
  return lower + triangle * (upper - lower)
# XML 七关节参考索引为 (joint2, joint3, joint4, joint5, joint6, joint7, joint1)。
ACTUATED_JOINT_NAMES: tuple[str, ...] = ("joint2", "joint3", "joint7", "joint1")
ACTUATED_REFERENCE_INDICES: tuple[int, ...] = (0, 1, 5, 6)


def actuated_nominal_targets(phase: float) -> tuple[float, ...]:
  """按实际执行器顺序提取安全七关节参考中的四个名义目标。"""
  target = Joint2FeasibleReference().targets(phase)
  return tuple(target[index] for index in ACTUATED_REFERENCE_INDICES)


def compose_bounded_residual_target(
  phase: float,
  residual: tuple[float, ...],
  *,
  residual_scale: float = 0.05,
) -> tuple[float, ...]:
  """将四维策略残差限幅后叠加到安全名义电机目标。"""
  if len(residual) != len(ACTUATED_JOINT_NAMES):
    raise ValueError("residual must contain four actuator values")
  if residual_scale < 0.0:
    raise ValueError("residual_scale must be non-negative")
  nominal = actuated_nominal_targets(phase)
  return tuple(
    value + max(-1.0, min(1.0, delta)) * residual_scale
    for value, delta in zip(nominal, residual, strict=True)
  )

def joint2_reference_torch(phase):
  """Torch 批量版七关节参考，输出形状为 (num_envs, 7)。"""
  import torch

  values = [torch.ones_like(phase), phase]
  for harmonic in range(1, 6):
    values.extend((torch.sin(harmonic * phase), torch.cos(harmonic * phase)))
  features = torch.stack(values, dim=1)
  coefficients = torch.tensor(_COEFFS, dtype=phase.dtype, device=phase.device)
  passive = features @ coefficients.T
  # passive 的列顺序为 J1,J3,J4,J5,J6,J7，重排为 XML joint2..joint7,joint1。
  return torch.cat((phase.unsqueeze(1), passive[:, 1:], passive[:, :1]), dim=1)