from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class ProtocolMode(str, Enum):
    mila = "mila"
    openai = "openai"
    anthropic = "anthropic"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MILA_",
        env_file=".env",
        case_sensitive=False,
    )

    # Model
    model: str = Field(
        "gemma-4-12b-it-fp4",
        description=(
            "Name of an installed model in the local Mila store. Also the identifier "
            "returned in API responses -- one name, so what a client sees is what is loaded."
        ),
    )
    context_length: Annotated[int, Field(gt=0)] | Literal["auto"] = Field(
        "auto",
        description=(
            "Tokens the model holds, prompt and reply together, passed to from_store(). 'auto' "
            "takes the longest the GPU can hold; the length actually loaded is loaded.context_length."
        ),
    )
    kv_cache: Literal["bf16", "fp8"] = Field(
        "bf16",
        description=(
            "How the model holds its context, passed to from_store(). 'fp8' holds it in half the memory, "
            "so 'auto' fits more of it."
        ),
    )
    device_index: int = Field(0, description="CUDA device ordinal.")

    # Generation defaults. Sampling has none here: a knob a request leaves out is the model's own recommended
    # setting, read from its manifest by the binding.
    default_max_new_tokens: int = Field(1024)

    # Streaming timeouts
    keepalive_interval: float = Field(15.0, description="Seconds between SSE keepalive pings during prefill.")
    decode_timeout: float = Field(30.0, description="Seconds to wait for each subsequent token during decode.")

    # Server
    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    log_level: str = Field("info")

    # Protocol
    protocol: ProtocolMode = Field(ProtocolMode.openai, description="API protocol to expose.")


settings = Settings()


def required_attribution_for(license_id: str) -> str:
    """
    The attribution a license requires be displayed wherever the model is presented.

    Llama 3.1 and 3.2 section 1.b.i require "Built with Llama" on a related website, user
    interface, blogpost, about page or product documentation. A server has no interface of
    its own, so the duty lands on the two places it does present the model: what it logs at
    startup, and what /v1/models returns.

    Empty for licenses with no display duty -- Apache 2.0 and MIT require the notice travel
    with the model, not that a consumer render one.
    """
    return "Built with Llama" if license_id.startswith("llama") else ""


@dataclass
class LoadedModel:
    """
    What the store record says about the model this process loaded.

    Deliberately not configuration. Variant, instruct-tuning and what the model can do are facts
    about the weights, and an environment variable that disagreed with them could only ever be
    wrong. ModelWorker fills this in at startup; the endpoints read it per request. Nothing here
    names a family: what varies by model is the model's chat protocol, which the worker holds.
    """

    name: str = settings.model
    variant: str = ""
    instruct: bool = True

    # Declared by the manifest, for the model cards a client reads before it drives the model.
    tools: bool = False
    thinking: bool = False

    # Lineage. Carried because a license that requires attribution requires it of whoever
    # presents the model, and serving it over an API is presenting it.
    base_model: str = ""
    license: str = ""

    # What the model was built to hold, as the load resolved it -- settings.context_length may
    # be "auto", so this is the number every prompt is bounded by and every model card reports.
    context_length: int = 0

    # What ran it: the library compiled into the binding, which a rebuild changes without a
    # reinstall, and the card, named by more than its CUDA ordinal -- that ordinal is not
    # nvidia-smi's, and two cards of one model differ only in their PCI address.
    mila_version: str = ""
    device: dict | None = None

    @property
    def attribution(self) -> str:
        return required_attribution_for(self.license)


loaded = LoadedModel()
