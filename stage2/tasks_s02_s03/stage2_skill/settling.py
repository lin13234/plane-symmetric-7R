"""Stage2 回放落地（settling）与最后一个连续 segment 的纯状态机。

评估器在回放中可能因 episode 终止而自动 reset。位移、周期与跟踪误差必须
只在“最后一个连续 episode”内计算，落地窗口指标也必须对应最后一个连续
episode，否则 reset 会把不同 episode 的数据混在一起。本模块把这段边界
跟踪抽成纯函数，输入每帧的 settled 与 done 标志，输出最后一个连续
episode 的落地窗口与 segment 起点，保证可单测、可复核。

done 标志的语义（与 evaluate_run.py 保持一致）：
  evaluate_run.py 在 ``env.step()`` 返回后才读取 asset，而自动 reset 已经
  在 ``step()`` 内部发生。因此 ``done=True`` 的帧对应的 asset 数据已经是
  post-reset 状态（新 episode 的首帧，一个悬空静止的“reset 静态帧”），
  done 标志本身对应“导致 reset 的那一步 transition”。为跳过这个静态帧，
  有效连续段从 done 帧的 ``step+1`` 起算；若这样得到的最后连续 episode
  不足两帧，则直接拒绝（见 track_settling 的 ValueError）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SettlingSummary:
  """最后一个连续 episode 的 settling 跟踪结果。"""

  segment_start: int
  """位移/周期/RMSE 的起点索引。

  settled=True 时为最后一个连续 episode 内首个 settled 帧；否则为该
  episode 首帧（跳过 reset 静态帧后的第一帧）。"""

  settling_start: int
  """最后一个连续 episode 落地窗口起点（= 该 episode 首帧，即跳过
  reset 静态帧后的第一帧）。"""

  settling_end: int
  """最后一个连续 episode 落地窗口终点。

  settled=True 时为首个 settled 帧；否则为该 episode 末帧。"""

  settled: bool
  """最后一个连续 episode 是否完成了 settling。"""

  reset_steps: tuple[int, ...]
  """全部 reset 事件帧索引（升序）。done=True 的帧本身已经是
  post-reset 状态。"""


def track_settling(settled: np.ndarray, dones: np.ndarray) -> SettlingSummary:
  """逐 episode 跟踪 settling，返回最后一个连续 episode 的结果。

  Args:
      settled: (N,) bool，每帧命令项是否已 settled。
      dones: (N,) bool，每帧是否触发自动 reset。done=True 的帧本身已经
          是 post-reset 状态（新 episode 的 reset 静态帧），因此有效连续
          段从该帧的 step+1 起算。

  规则：
    - done 帧对应的 reset 静态帧被跳过，下一个 episode 的有效帧从该帧的
      step+1 开始；
    - 每个 episode 独立跟踪 settling，reset 时清空上一 episode 状态；
    - 最后一个连续 episode 首个 settled 帧即 segment_start；若该 episode
      未 settled，segment_start 为该 episode 首帧；
    - 未 settled 时 settling_end 取该 episode 末帧索引。

  Raises:
      ValueError: settled 与 dones 不是等长的一维数组，或轨迹为空，或
          最后一个连续 episode 少于 2 个可记录帧（回放太短或最后一次
          reset 后样本不足）。
  """
  settled_arr = np.asarray(settled, dtype=bool)
  dones_arr = np.asarray(dones, dtype=bool)
  if settled_arr.ndim != 1 or dones_arr.ndim != 1:
    raise ValueError("settled and dones must be 1-D arrays")
  if settled_arr.shape != dones_arr.shape:
    raise ValueError("settled and dones must have the same length")
  n = settled_arr.shape[0]
  if n == 0:
    raise ValueError("empty trajectory")

  reset_steps: list[int] = []
  episode_start = 0
  for step in range(n):
    if dones_arr[step]:
      reset_steps.append(step)
      # done 帧本身是 post-reset 静态帧，跳过它，从 step+1 起算。
      episode_start = step + 1

  # 最后一个连续 episode 的有效帧数：从 episode_start（含）到 n-1。
  # 不足 2 帧时无法计算位移/周期/RMSE，且绝不能为凑帧而向前夹进
  # 上一个 episode，因此直接拒绝。
  if n - episode_start < 2:
    raise ValueError(
      f"last consecutive episode has only {n - episode_start} recordable "
      "frame(s); replay too short or insufficient samples after the last "
      "reset, increase --steps"
    )

  # 最后一个连续 episode 内找首个 settled 帧。
  settle_idx: int | None = None
  for step in range(episode_start, n):
    if settled_arr[step]:
      settle_idx = step
      break

  if settle_idx is not None:
    segment_start = settle_idx
    settling_end = settle_idx
    did_settle = True
  else:
    segment_start = episode_start
    settling_end = n - 1
    did_settle = False

  return SettlingSummary(
    segment_start=int(segment_start),
    settling_start=int(episode_start),
    settling_end=int(settling_end),
    settled=did_settle,
    reset_steps=tuple(reset_steps),
  )


def require_settled(summary: SettlingSummary) -> None:
  """评估器保护：未完成 settling 的最后一个 episode 拒绝计算运动指标。

  落地瞬态不是技能运动数据；位移、周期与 RMSE 只允许在 settled 段内计算。
  ``settled=False`` 说明回放结束时最后一个连续 episode 仍在落地窗口内，
  继续计算会把落地瞬态混进指标。调用方应提示增加 ``--steps``，让最后一个
  episode 完整经历落地窗口后再采集技能运动。

  Args:
      summary: track_settling 的返回结果。

  Raises:
      ValueError: summary.settled 为 False。
  """
  if not summary.settled:
    raise ValueError(
      "last consecutive episode did not finish settling; motion metrics "
      "would be contaminated by the landing transient, increase --steps "
      "so the final episode completes its settling window"
    )
