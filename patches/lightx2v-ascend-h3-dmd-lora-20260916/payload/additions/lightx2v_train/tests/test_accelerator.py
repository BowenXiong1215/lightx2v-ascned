import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_MODULE_PATH = Path(__file__).parents[1] / "lightx2v_train" / "runtime" / "accelerator.py"
_SPEC = importlib.util.spec_from_file_location("lightx2v_train_accelerator_test_target", _MODULE_PATH)
accelerator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(accelerator)


@pytest.fixture(autouse=True)
def reset_accelerator():
    old = accelerator._ACCELERATOR_TYPE
    accelerator._ACCELERATOR_TYPE = None
    yield
    accelerator._ACCELERATOR_TYPE = old


def test_hccl_selects_npu():
    assert accelerator.resolve_accelerator_type({"distributed": {"backend": "hccl"}}) == "npu"


def test_explicit_cpu_overrides_backend():
    config = {"distributed": {"backend": "hccl", "device_type": "cpu"}}
    assert accelerator.resolve_accelerator_type(config) == "cpu"


def test_ascend_alias_is_normalized():
    config = {"distributed": {"device_type": "ascend_npu"}}
    assert accelerator.resolve_accelerator_type(config) == "npu"


def test_initialize_npu_imports_runtime_and_sets_local_rank(monkeypatch):
    calls = []
    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        set_device=lambda rank: calls.append(rank),
        current_device=lambda: calls[-1],
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    assert accelerator.initialize_accelerator({"distributed": {"backend": "hccl"}}, local_rank=3) == "npu"
    assert calls == [3]
    assert accelerator.get_accelerator_type() == "npu"


def test_invalid_accelerator_is_rejected():
    with pytest.raises(ValueError, match="Unsupported distributed.device_type"):
        accelerator.resolve_accelerator_type({"distributed": {"device_type": "tpu"}})
