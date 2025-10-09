# normalize_torch.py
from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional

import numpy as np
import pydantic
import numpydantic
import torch


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None
    q99: numpydantic.NDArray | None = None


class RunningStatsTorchParallel:
    """
    병렬(스태킹) 업데이트 + 세그먼트 경계에서만 리빈 수행.
    - 에지 규칙과 리빈 타이밍은 원본과 동일하게 재현.
    - 세그먼트(범위 확장 없는 구간) 내부는 여러 배치를 GPU에 쌓아 한 번에 히스토그램 갱신.

    사용 절차:
      1) ingest_batch(x) 를 데이터 순서대로 반복 호출
      2) finalize() 호출 -> NormStats 반환
    """

    def __init__(
        self,
        num_quantile_bins: int = 5000,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        pack_n_batches: int = 8,           # 세그먼트 내에서 몇 개 배치를 쌓아 한 번에 처리할지
        pack_max_elems: Optional[int] = None,  # 누적 샘플 수 상한(메모리 안전장치). None이면 미사용
        deterministic: bool = False,
        eps: float = 1e-10,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.B = int(num_quantile_bins)
        self.eps = float(eps)

        if deterministic:
            torch.use_deterministic_algorithms(True)

        # 러닝 통계
        self.count: int = 0
        self.sum_: Optional[torch.Tensor] = None     # (D,)
        self.sumsq_: Optional[torch.Tensor] = None   # (D,)

        # 전역 범위(데이터 전체의 최소/최대)
        self.gmin: Optional[torch.Tensor] = None     # (D,)
        self.gmax: Optional[torch.Tensor] = None     # (D,)

        # 첫 배치 범위(초기 에지에 eps 포함 여부 판단 및 초기 에지 구성)
        self.first_min: Optional[torch.Tensor] = None
        self.first_max: Optional[torch.Tensor] = None
        self.seen_first: bool = False

        # 히스토그램 및 에지(세그먼트 현재 에지)
        self.hist: Optional[torch.Tensor] = None     # (D, B), float64
        self.edge_start: Optional[torch.Tensor] = None  # (D,)
        self.edge_delta: Optional[torch.Tensor] = None  # (D,)

        # 패킹 버퍼
        self.pack: List[torch.Tensor] = []  # 각 텐서는 (Ni, D)
        self.pack_batches: int = 0
        self.pack_elems: int = 0
        self.pack_n_batches = int(pack_n_batches)
        self.pack_max_elems = pack_max_elems

        # 특성 차원
        self.D: Optional[int] = None

    # ---------- 내부 유틸 ----------

    def _to_device_2d(self, x) -> torch.Tensor:
        t = x
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=self.dtype)
        else:
            if t.dtype != self.dtype:
                t = t.to(self.dtype)
        # reshape to [N, D]
        t = t.reshape(-1, t.shape[-1])
        # 비동기 복사 (pin-memory일 때 효과)
        if t.device != self.device:
            t = t.to(self.device, non_blocking=True)
        return t

    def _init_from_first_batch(self, t: torch.Tensor):
        assert t.ndim == 2
        N, D = t.shape
        self.D = D

        # 러닝 통계 초기화
        vmin = t.min(dim=0).values
        vmax = t.max(dim=0).values
        self.first_min = vmin.clone()
        self.first_max = vmax.clone()
        self.gmin = vmin.clone()
        self.gmax = vmax.clone()

        self.sum_ = t.sum(dim=0)
        self.sumsq_ = (t * t).sum(dim=0)
        self.count = N

        # 초기 에지: [min - eps, max + eps] (원본과 동일)
        start = vmin - self.eps
        end = vmax + self.eps
        delta = (end - start) / self.B

        self.edge_start = start
        self.edge_delta = delta

        self.hist = torch.zeros((D, self.B), dtype=self.dtype, device=self.device)

        # 첫 배치는 일단 패킹 버퍼에 쌓아둠(동일 에지 구간이므로 합산해도 동일)
        self.pack.append(t)
        self.pack_batches = 1
        self.pack_elems = N
        self.seen_first = True

    def _flush_pack(self):
        """현재 에지에 대해 패킹된 배치들을 한 번에 히스토그램에 누적."""
        if self.pack_batches == 0:
            return
        assert self.hist is not None and self.edge_start is not None and self.edge_delta is not None
        X = torch.cat(self.pack, dim=0)  # (M, D)
        # delta==0인 차원은 모든 값이 0번 bin으로 가도록 처리
        delta = self.edge_delta
        delta_safe = torch.where(delta == 0, torch.ones_like(delta), delta)
        idx = torch.floor((X - self.edge_start) / delta_safe).clamp_(0, self.B - 1).to(torch.long)  # (M, D)
        idxT = idx.transpose(0, 1)  # (D, M)
        ones = torch.ones_like(idxT, dtype=self.hist.dtype)
        self.hist.scatter_add_(dim=1, index=idxT, src=ones)

        # 버퍼 초기화
        self.pack.clear()
        self.pack_batches = 0
        self.pack_elems = 0

    def _rebin_left_edge(self, old_start: torch.Tensor, old_delta: torch.Tensor, new_start: torch.Tensor, new_delta: torch.Tensor):
        """
        원본 _adjust_histograms와 동일한 규칙:
          - 옛 에지의 '좌측 에지들'(old_left_edges = old_start + k*old_delta)을
            새 에지에 np.histogram(...) with weights=old_hist 로 재분배.
          - 여기서는 완전 벡터화된 scatter_add_로 수행.
        """
        assert self.hist is not None
        D, B = self.hist.shape

        k = torch.arange(B, device=self.device, dtype=self.dtype)  # (B,)
        old_left_edges = old_start.unsqueeze(1) + k.unsqueeze(0) * old_delta.unsqueeze(1)  # (D, B)

        # new_delta==0 보호
        new_delta_safe = torch.where(new_delta == 0, torch.ones_like(new_delta), new_delta)
        idx = torch.floor((old_left_edges - new_start.unsqueeze(1)) / new_delta_safe.unsqueeze(1))
        idx = idx.clamp_(0, B - 1).to(torch.long)  # (D, B)

        new_hist = torch.zeros_like(self.hist)
        new_hist.scatter_add_(dim=1, index=idx, src=self.hist)
        self.hist = new_hist

    def _expand_edges_if_needed(self, bmin: torch.Tensor, bmax: torch.Tensor):
        """
        이 배치가 전역 범위를 확장하면:
          1) 현재 패킹 버퍼를 현 에지로 플러시
          2) 히스토그램을 새 에지로 리빈(원본 타이밍과 동일: rebin이 '이 배치를 더하기 전에' 발생)
          3) 전역 범위를 갱신하고 새 에지를 설정
        """
        assert self.gmin is not None and self.gmax is not None
        need_expand = (bmin < self.gmin).any() or (bmax > self.gmax).any()
        if not need_expand:
            return

        # 1) 현재까지 쌓인(이 배치 제외) 것들을 현 에지로 먼저 누적
        self._flush_pack()

        # 2) 새 전역 범위
        new_min = torch.minimum(self.gmin, bmin)
        new_max = torch.maximum(self.gmax, bmax)

        # 3) 리빈: 새 에지는 linspace(new_min, new_max, B+1) => start=new_min, delta=(new_max-new_min)/B
        old_start, old_delta = self.edge_start, self.edge_delta
        new_start = new_min
        new_delta = (new_max - new_start) / self.B

        # 기존 히스토그램을 새 에지로 재분배
        self._rebin_left_edge(old_start, old_delta, new_start, new_delta)

        # 상태 갱신
        self.gmin, self.gmax = new_min, new_max
        self.edge_start, self.edge_delta = new_start, new_delta

    # ---------- 퍼블릭 API ----------

    @torch.no_grad()
    def ingest_batch(self, x):
        """
        데이터 순서대로 호출. 내부적으로 자동 패킹/플러시/리빈 수행.
        """
        t = self._to_device_2d(x)  # (N, D)

        if not self.seen_first:
            self._init_from_first_batch(t)
            return

        # 러닝 통계 누적 (sum, sumsq, count) — 순서 불변
        assert self.sum_ is not None and self.sumsq_ is not None
        self.sum_ += t.sum(dim=0)
        self.sumsq_ += (t * t).sum(dim=0)
        self.count += t.shape[0]

        # 이 배치가 범위를 확장하는지 먼저 확인(원본 타이밍과 동일하게 rebin이 이 배치 누적 "전" 수행)
        bmin = t.min(dim=0).values
        bmax = t.max(dim=0).values
        self._expand_edges_if_needed(bmin, bmax)

        # 패킹 버퍼에 추가(세그먼트 내에서는 에지가 고정이므로 여러 배치를 합쳐도 동일)
        self.pack.append(t)
        self.pack_batches += 1
        self.pack_elems += t.shape[0]

        # 패킹 임계치 도달 시 즉시 플러시
        if (self.pack_batches >= self.pack_n_batches) or (
            self.pack_max_elems is not None and self.pack_elems >= self.pack_max_elems
        ):
            self._flush_pack()

    @torch.no_grad()
    def finalize(self) -> NormStats:
        """
        남은 패킹 버퍼를 플러시하고 평균/표준편차/분위를 계산하여 반환.
        """
        if not self.seen_first or self.count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        # 마지막 플러시
        self._flush_pack()

        # 평균/표준편차
        mean = (self.sum_ / float(self.count)).clone()
        var = (self.sumsq_ / float(self.count)) - mean * mean
        var.clamp_(min=0.0)
        std = torch.sqrt(var)

        # 분위 계산 (원본과 동일: 누적합에서 타깃 카운트의 '좌측' 인덱스)
        assert self.hist is not None and self.edge_start is not None and self.edge_delta is not None
        cumsum = torch.cumsum(self.hist, dim=1)
        total = float(self.count)

        def q_to_val(q: float) -> torch.Tensor:
            target = self.hist.new_tensor(q * total)
            idx = torch.sum(cumsum < target, dim=1)  # (D,)
            return self.edge_start + self.edge_delta * idx.to(self.edge_start.dtype)

        q01 = q_to_val(0.01)
        q99 = q_to_val(0.99)

        # NumPy로 변환하여 원본 JSON 스키마에 맞춤
        return NormStats(
            mean=mean.detach().cpu().numpy(),
            std=std.detach().cpu().numpy(),
            q01=q01.detach().cpu().numpy(),
            q99=q99.detach().cpu().numpy(),
        )


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: Dict[str, NormStats]


def serialize_json(norm_stats: Dict[str, NormStats]) -> str:
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> Dict[str, NormStats]:
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: Dict[str, NormStats]) -> None:
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> Dict[str, NormStats]:
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())