"""Unit tests for ror.training helpers (the full loop is in test_pipeline_tiny)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_student_trains_on_one_gpu_when_several_are_visible(tmp_path):
    trl = pytest.importorskip("trl")
    from ror.training import _single_device

    args = trl.SFTConfig(output_dir=str(tmp_path), use_cpu=True, report_to="none",
                         per_device_train_batch_size=2)
    args._n_gpu = 2                       # what TrainingArguments reports on Kaggle's T4x2
    assert args.train_batch_size == 4     # the Trainer would wrap in DataParallel, 2x batch
    _single_device(args)
    assert args.n_gpu == 1 and args.train_batch_size == 2


def test_single_gpu_or_cpu_args_are_left_alone(tmp_path):
    trl = pytest.importorskip("trl")
    from ror.training import _single_device

    args = trl.SFTConfig(output_dir=str(tmp_path), use_cpu=True, report_to="none")
    before = args.n_gpu
    _single_device(args)
    assert args.n_gpu == before
