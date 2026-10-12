"""
mila-server's command line: serve with no command, list and install models with one.

The binding's store and planner are replaced by fakes, so the verdict words and the install loop run
with no store on disk and no GPU. What the planner answers is the library's and is tested there.
"""
from types import SimpleNamespace

import mila
import pytest

from mila_llm_server import cli
from mila_llm_server.config import settings


def record(name, complete=True, size=2 * 1024 ** 3, license="apache-2.0"):
    return SimpleNamespace(name=name, complete=complete, bytes_on_disk=size, license=license)


class FakeStore:
    def __init__(self, records=(), published=(), failing=()):
        self.records = list(records)
        self.published = list(published)
        self.failing = set(failing)
        self.pulled = []

    def list(self):
        return list(self.records)

    def list_hub_models(self, owner):
        return list(self.published)

    def pull(self, name, owner, transport=None):
        self.pulled.append((name, owner))

        if name in self.failing:
            raise RuntimeError(f"no published model named '{name}'")

        transport("https://example/weights", {}, lambda chunk: True)

        return record(name)


@pytest.fixture
def fake(monkeypatch):
    """A store, the planner's answers per name, and one card at the configured index."""
    state = SimpleNamespace(
        store=FakeStore(),
        stored={},
        fitting={},
        devices=[SimpleNamespace(index=0, name="NVIDIA GeForce RTX 4070")],
    )

    monkeypatch.setattr(mila, "ModelStore", lambda: state.store)
    monkeypatch.setattr(mila, "initialize", lambda **_: None)
    monkeypatch.setattr(mila, "cuda_devices", lambda: state.devices)
    monkeypatch.setattr(mila, "default_hub_owner", lambda: "mila-models")
    monkeypatch.setattr(mila, "stored_weight_quantization", lambda name: state.stored[name])
    monkeypatch.setattr(mila, "fitting_weight_quantizations", lambda name, device_index: state.fitting[name])
    monkeypatch.setattr(mila, "http_transport", lambda url, headers, writer: writer(b"x" * 1024))
    monkeypatch.setattr(settings, "device_index", 0)

    return state


def test_models_reports_the_planners_verdict_in_the_words_a_load_takes(fake, capsys):
    fake.store.records = [
        record("gemma-4-12b-it-fp4"),
        record("Llama-3.1-8B-Instruct-bf16"),
        record("Qwen3.8-27B-bf16"),
        record("broken-package"),
        record("half-installed", complete=False),
    ]
    fake.stored = {
        "gemma-4-12b-it-fp4": "fp4",
        "Llama-3.1-8B-Instruct-bf16": "bf16",
        "Qwen3.8-27B-bf16": "bf16",
        "broken-package": "fp4",
    }
    fake.fitting = {
        "gemma-4-12b-it-fp4": ["fp4"],
        # Reference weights that do not fit as stored, and do once quantized as they load.
        "Llama-3.1-8B-Instruct-bf16": ["fp8", "fp4"],
        "Qwen3.8-27B-bf16": [],
        "broken-package": None,
    }

    assert cli.main(["models"]) == 0

    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[1:]}

    assert lines["gemma-4-12b-it-fp4"].endswith("yes")
    assert lines["Llama-3.1-8B-Instruct-bf16"].endswith("yes, at fp8 or fp4")
    assert lines["Qwen3.8-27B-bf16"].endswith("no")
    assert lines["broken-package"].endswith("unknown")
    assert lines["half-installed"].endswith("-  (incomplete)")


def test_models_asks_about_the_card_the_server_would_serve_on(fake, monkeypatch, capsys):
    asked = []
    fake.store.records = [record("gemma-4-12b-it-fp4")]
    fake.stored = {"gemma-4-12b-it-fp4": "fp4"}
    fake.devices = [SimpleNamespace(index=0, name="card zero"), SimpleNamespace(index=1, name="card one")]

    def fitting(name, device_index):
        asked.append(device_index)
        return ["fp4"]

    monkeypatch.setattr(mila, "fitting_weight_quantizations", fitting)
    monkeypatch.setattr(settings, "device_index", 1)

    cli.main(["models"])

    assert asked == [1]
    assert "card one" in capsys.readouterr().out


def test_models_without_the_configured_card_says_so_and_plans_nothing(fake, monkeypatch, capsys):
    fake.store.records = [record("gemma-4-12b-it-fp4")]
    fake.devices = []

    assert cli.main(["models"]) == 0

    output = capsys.readouterr().out
    assert "No CUDA device 0" in output
    assert "yes" not in output


def test_models_on_an_empty_store_names_the_commands_that_fill_it(fake, capsys):
    assert cli.main(["models"]) == 0

    output = capsys.readouterr().out
    assert "mila-server models --online" in output
    assert "mila-server install <name>" in output


def test_models_online_lists_what_is_published(fake, capsys):
    fake.store.published = [
        SimpleNamespace(repository="gemma-4-12b-it-fp4", gated=False),
        SimpleNamespace(repository="Llama-3.2-3B-Instruct-fp4", gated=True),
    ]

    assert cli.main(["models", "--online"]) == 0

    output = capsys.readouterr().out
    assert "gemma-4-12b-it-fp4" in output
    assert "Llama-3.2-3B-Instruct-fp4  (gated)" in output


def test_install_pulls_each_name_from_the_default_owner_and_reports_the_license(fake, capsys):
    assert cli.main(["install", "gemma-4-12b-it-fp4", "Qwen3.8-27B-fp4"]) == 0

    assert fake.store.pulled == [("gemma-4-12b-it-fp4", "mila-models"), ("Qwen3.8-27B-fp4", "mila-models")]

    output = capsys.readouterr().out
    assert "Installed gemma-4-12b-it-fp4" in output
    assert "License: apache-2.0" in output


def test_install_carries_on_past_a_failure_and_exits_nonzero(fake, capsys):
    fake.store.failing = {"misspelled"}

    assert cli.main(["install", "misspelled", "gemma-4-12b-it-fp4"]) == 1

    captured = capsys.readouterr()
    assert "Could not install 'misspelled'" in captured.err
    assert "Installed gemma-4-12b-it-fp4" in captured.out


def test_install_without_a_name_is_refused(fake):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["install"])

    assert excinfo.value.code == 2


def test_no_command_serves(monkeypatch):
    from mila_llm_server import app

    served = []
    monkeypatch.setattr(app, "serve", lambda: served.append(True))

    assert cli.main([]) == 0
    assert served == [True]


def test_an_unknown_argument_is_refused_rather_than_dropped(monkeypatch):
    from mila_llm_server import app

    monkeypatch.setattr(app, "serve", lambda: pytest.fail("served despite an unknown argument"))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--port", "9000"])

    assert excinfo.value.code == 2
