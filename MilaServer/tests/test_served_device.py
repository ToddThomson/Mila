"""
/v1/models names the library and the card that served the model.

A client that records where a measurement came from reads both from the model card. The CUDA
ordinal alone does not name a card -- it is not nvidia-smi's index, and two cards of one model
share a name -- so the record carries the PCI address in nvidia-smi's form. Stand-in devices
replace the runtime's list, so this runs with no GPU.
"""
import dataclasses
from types import SimpleNamespace

import pytest

from mila_llm_server import model_worker
from mila_llm_server.config import loaded
from mila_llm_server.protocols.openai.models import OpenAIModelsAdapter


def _device(index, name, bus, capability=(8, 9)):
    return SimpleNamespace(
        index=index,
        name=name,
        compute_capability=capability,
        total_memory_bytes=12 << 30,
        pci_domain=0,
        pci_bus=bus,
        pci_device=0,
    )


@pytest.fixture
def two_cards(monkeypatch):
    devices = [_device(0, "Card A", 6), _device(1, "Card B", 1, (12, 0))]
    monkeypatch.setattr(model_worker.mila, "cuda_devices", lambda: devices, raising=False)


def test_the_record_is_the_card_at_the_cuda_ordinal_not_the_first_listed(two_cards):
    record = model_worker._device_record(1)

    assert record["name"] == "Card B"
    assert record["compute_capability"] == "12.0"
    assert record["pci_bus_id"] == "00000000:01:00.0"


def test_an_ordinal_the_runtime_does_not_list_has_no_record(two_cards):
    assert model_worker._device_record(2) is None


def test_the_model_card_carries_the_version_and_the_device(two_cards):
    before = dataclasses.replace(loaded)
    loaded.mila_version = "0.21.0-dev+54"
    loaded.device = model_worker._device_record(0)

    try:
        card = OpenAIModelsAdapter().format_models_response()["data"][0]
    finally:
        for field in dataclasses.fields(before):
            setattr(loaded, field.name, getattr(before, field.name))

    assert card["mila_version"] == "0.21.0-dev+54"
    assert card["device"]["pci_bus_id"] == "00000000:06:00.0"
