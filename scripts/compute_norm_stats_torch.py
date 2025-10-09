# compute_norm_stats_torch.py
"""
Torch-accelerated, segment-parallel computation of normalization statistics.

핵심:
- 각 키(state, actions)에 대해 RunningStatsTorchParallel을 사용
- 세그먼트(범위 확장이 없는 구간) 내에서 배치들을 GPU에 '여러 개 쌓아' 한 번에 히스토그램 누적
- 세그먼트 경계에서만 리빈(원본 타이밍과 동일)

실행 예:
  CPU(완전 동일성 최우선):
    python compute_norm_stats_torch.py <CONFIG> --device cpu --dtype float64

  GPU(속도 최우선):
    python compute_norm_stats_torch.py <CONFIG> --device cuda --dtype float64 --pack-n-batches 16
"""

from __future__ import annotations

import numpy as np
import tqdm
import tyro
import torch

import openpi.models.model as _model
import openpi.shared.normalize_torch as normalize  # 위 파일 이름이 normalize_torch.py 라고 가정
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        # 만약 내부 구현이 PyTorch DataLoader를 사용한다면 pin_memory=True가 좋습니다.
        # 여기서는 openpi 내부 구현이라 인자 노출이 없을 수 있습니다.
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,

    # Torch 런타임
    device: str | None = None,           # None이면 자동 선택(cuda -> mps -> cpu)
    dtype: str = "float64",              # "float64"(권장, 원본과 동일) 또는 "float32"
    deterministic: bool = False,

    # 히스토그램/패킹 파라미터
    num_quantile_bins: int = 5000,
    pack_n_batches: int = 8,             # 세그먼트 내에서 한 번에 쌓을 배치 수
    pack_max_elems: int | None = None,   # 누적 샘플 수 상한 (GPU 메모리 제한용)
):
    # 디바이스/정밀도
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # Apple MPS
            device = "mps"
        else:
            device = "cpu"
    device_obj = torch.device(device)
    dtype_obj = torch.float64 if dtype == "float64" else torch.float32
    if deterministic:
        torch.use_deterministic_algorithms(True)

    print(f"[Torch] device={device_obj}, dtype={dtype_obj}, cuda_available={torch.cuda.is_available()}")

    # 설정 로드
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # 데이터로더
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    # 키별 병렬 러닝 통계기
    keys = ["state", "actions"]
    stats = {
        key: normalize.RunningStatsTorchParallel(
            num_quantile_bins=num_quantile_bins,
            device=device_obj,
            dtype=dtype_obj,
            pack_n_batches=pack_n_batches,
            pack_max_elems=pack_max_elems,
            deterministic=deterministic,
        )
        for key in keys
    }

    # 스트리밍 1패스: 세그먼트 내부는 패킹/병렬, 경계에서만 리빈
    for i, batch in enumerate(tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats (segment-parallel)")):
        for key in keys:
            # batch[key]는 numpy 또는 torch 텐서일 수 있음
            stats[key].ingest_batch(batch[key])

    # 마무리 및 저장
    norm_stats = {key: stat.finalize() for key, stat in stats.items()}

    output_path = config.assets_dirs / (data_config.repo_id or "unknown_repo")
    print("stats:", norm_stats)  # 프로세스 중단 대비
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)