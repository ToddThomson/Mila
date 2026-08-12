"""
ModelWorker._load: store lookup, the refuse-to-pull message, and the architecture guard.

This is the only place a startup can fail, and it had no coverage. A fake store stands in
for mila.ModelStore so all three paths run with no store on disk and no GPU -- the binding
is imported (it is a dependency) but nothing is initialized and no weights are touched.
"""
import dataclasses

import mila
import pytest

from mila_llm_server import model_worker
from mila_llm_server.config import ModelFamily, loaded, settings
from mila_llm_server.model_worker import ModelWorker, _family_of


class FakeRecord:
    """The fields ModelWorker reads off a mila.StoredModel."""

    def __init__(
        self,
        name="gemma-4-12b-it-fp4",
        architecture="gemma",
        variant="fp4",
        instruct=True,
        base_model="google/gemma-4-12b-it",
        license="apache-2.0",
    ):
        self.name = name
        self.architecture = architecture
        self.variant = variant
        self.instruct = instruct
        self.base_model = base_model
        self.license = license


class FakeStore:
    """Resolves by exact name only; the real store's case-insensitivity is exercised by
    handing back a record whose name differs from what was asked for."""

    root = "/fake/store"

    def __init__(self, records=()):
        self._records = list(records)

    def locate(self, name):
        return next((r for r in self._records if r.name.lower() == name.lower()), None)

    def list(self):
        return list(self._records)


@pytest.fixture
def store(monkeypatch):
    """Installs a fake store and stubs the loaders, and restores config.loaded after.

    config.loaded is a module-level singleton the adapters read per request, so a test
    that let a mutation escape would change what every later test sees.
    """
    before = dataclasses.replace(loaded)

    holder = {"store": FakeStore(), "loaded_name": None, "session": None}

    monkeypatch.setattr(mila, "ModelStore", lambda: holder["store"])
    monkeypatch.setattr(
        mila.BpeTokenizer, "from_store", staticmethod(lambda name: f"tokenizer:{name}")
    )

    def make_session(label):
        def from_store(name, context_length, device_index):
            holder["session"] = label
            holder["loaded_name"] = name
            return f"model:{name}"

        return from_store

    monkeypatch.setattr(mila.GemmaModel, "from_store", staticmethod(make_session("gemma")))
    monkeypatch.setattr(mila.LlamaModel, "from_store", staticmethod(make_session("llama")))

    yield holder

    for field in dataclasses.fields(before):
        setattr(loaded, field.name, getattr(before, field.name))


def test_missing_model_names_what_is_installed(store, monkeypatch):
    store["store"] = FakeStore([FakeRecord(name="Llama-3.2-3B-Instruct-fp4", architecture="llama")])
    monkeypatch.setattr(settings, "model", "gemma-4-12b-it-fp4")

    with pytest.raises(RuntimeError) as excinfo:
        ModelWorker()._load()

    message = str(excinfo.value)
    assert "gemma-4-12b-it-fp4" in message
    assert "Llama-3.2-3B-Instruct-fp4" in message
    # The refusal has to say a download is not coming, or the reading is "it failed".
    assert "loads only what is already installed" in message


def test_empty_store_says_nothing_rather_than_an_empty_list(store, monkeypatch):
    store["store"] = FakeStore([])
    monkeypatch.setattr(settings, "model", "gemma-4-12b-it-fp4")

    with pytest.raises(RuntimeError) as excinfo:
        ModelWorker()._load()

    assert "Installed: nothing." in str(excinfo.value)


def test_unservable_architecture_is_refused_at_load(store, monkeypatch):
    store["store"] = FakeStore([FakeRecord(name="gpt2-small", architecture="gpt2")])
    monkeypatch.setattr(settings, "model", "gpt2-small")

    with pytest.raises(RuntimeError) as excinfo:
        ModelWorker()._load()

    message = str(excinfo.value)
    assert "gpt2" in message
    # Naming the supported set is what makes the refusal actionable.
    assert "gemma" in message and "llama" in message


def test_family_guard_accepts_both_served_architectures():
    assert _family_of(FakeRecord(architecture="gemma")) is ModelFamily.gemma
    assert _family_of(FakeRecord(architecture="llama")) is ModelFamily.llama


def test_load_fills_loaded_from_the_record_not_from_settings(store, monkeypatch):
    # The store matches case-insensitively, so the configured spelling and the record's
    # own name can differ. What a client sees must be the record's.
    record = FakeRecord(
        name="Llama-3.1-8B-Instruct-fp4",
        architecture="llama",
        variant="fp4",
        instruct=True,
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        license="llama3.1",
    )
    store["store"] = FakeStore([record])
    monkeypatch.setattr(settings, "model", "llama-3.1-8b-instruct-fp4")

    ModelWorker()._load()

    assert loaded.name == "Llama-3.1-8B-Instruct-fp4"
    assert loaded.family is ModelFamily.llama
    assert loaded.variant == "fp4"
    assert loaded.instruct is True
    assert loaded.base_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert loaded.license == "llama3.1"
    # Section 1.b.i: serving a Llama over an API is presenting it.
    assert loaded.attribution == "Built with Llama"
    assert store["session"] == "llama"
    assert store["loaded_name"] == "Llama-3.1-8B-Instruct-fp4"


def test_gemma_record_selects_the_gemma_session_and_asks_no_attribution(store, monkeypatch):
    store["store"] = FakeStore([FakeRecord()])
    monkeypatch.setattr(settings, "model", "gemma-4-12b-it-fp4")

    ModelWorker()._load()

    assert loaded.family is ModelFamily.gemma
    assert store["session"] == "gemma"
    # Apache 2.0 requires the notice travel with the artifact, not that a server render one.
    assert loaded.attribution == ""


def test_reported_identifier_is_the_loaded_name_not_the_configured_one(store, monkeypatch):
    # The regression this pins: the identifier was a module constant bound to
    # settings.model at import, before the worker had resolved anything, so a
    # case-insensitive store match reported a name that was never loaded.
    from mila_llm_server.protocols.openai.models import OpenAIModelsAdapter

    store["store"] = FakeStore(
        [FakeRecord(name="Llama-3.1-8B-Instruct-fp4", architecture="llama", license="llama3.1")]
    )
    monkeypatch.setattr(settings, "model", "llama-3.1-8b-instruct-fp4")

    ModelWorker()._load()
    card = OpenAIModelsAdapter().format_models_response()["data"][0]

    assert card["id"] == "Llama-3.1-8B-Instruct-fp4"
    assert card["slug"] == "Llama-3.1-8B-Instruct-fp4"
    assert card["display_name"] == "Llama-3.1-8B-Instruct-fp4"
    # Section 1.b.i again: this endpoint is where a client presents the model to a person,
    # and it served no lineage at all until the fields were added to the live card.
    assert card["attribution"] == "Built with Llama"
    assert card["license"] == "llama3.1"


def test_anthropic_serves_models_with_attribution(store, monkeypatch):
    # The Anthropic protocol had no /v1/models at all, so a licence that requires displayed
    # attribution was discharged by a startup log line and nothing else.
    from mila_llm_server.protocols.anthropic import AnthropicAdapter

    store["store"] = FakeStore(
        [FakeRecord(name="Llama-3.1-8B-Instruct-fp4", architecture="llama", license="llama3.1")]
    )
    monkeypatch.setattr(settings, "model", "llama-3.1-8b-instruct-fp4")

    ModelWorker()._load()
    payload = AnthropicAdapter().format_models_response()
    card = payload["data"][0]

    assert card["type"] == "model"
    assert card["id"] == "Llama-3.1-8B-Instruct-fp4"
    assert card["attribution"] == "Built with Llama"
    # Anthropic's pagination envelope, not OpenAI's {"object": "list"}.
    assert payload["has_more"] is False


def test_both_adapters_are_models_capable():
    """The factory registers /v1/models only for a ModelsCapable adapter, so this is the
    property that decides whether the endpoint exists at all."""
    from mila_llm_server.protocols.anthropic import AnthropicAdapter
    from mila_llm_server.protocols.base import ModelsCapable
    from mila_llm_server.protocols.openai import OpenAIAdapter

    assert issubclass(OpenAIAdapter, ModelsCapable)
    assert issubclass(AnthropicAdapter, ModelsCapable)
    assert AnthropicAdapter().models_path == "/v1/models"


def test_load_passes_the_configured_context_and_device(store, monkeypatch):
    captured = {}

    def from_store(name, context_length, device_index):
        captured["context_length"] = context_length
        captured["device_index"] = device_index
        return "model"

    monkeypatch.setattr(mila.GemmaModel, "from_store", staticmethod(from_store))
    store["store"] = FakeStore([FakeRecord()])
    monkeypatch.setattr(settings, "model", "gemma-4-12b-it-fp4")
    monkeypatch.setattr(settings, "context_length", 8192)
    monkeypatch.setattr(settings, "device_index", 1)

    ModelWorker()._load()

    assert captured == {"context_length": 8192, "device_index": 1}
