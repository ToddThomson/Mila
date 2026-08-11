from dataclasses import dataclass
from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class ProtocolMode(str, Enum):
    mila = "mila"
    openai = "openai"
    anthropic = "anthropic"


class ModelFamily(str, Enum):
    llama = "llama"
    gemma = "gemma"


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
    context_length: int = Field(4096, description="Maximum sequence length passed to from_store().")
    device_index: int = Field(0, description="CUDA device ordinal.")

    # Generation defaults
    default_max_new_tokens: int = Field(1024)
    default_temperature: float = Field(0.6)
    default_top_k: int = Field(40)
    default_top_p: float = Field(0.9)

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
    with the artifact, not that a consumer render one.
    """
    return "Built with Llama" if license_id.startswith("llama") else ""


@dataclass
class LoadedModel:
    """
    What the store record says about the model this process loaded.

    Deliberately not configuration. Family, variant and instruct-tuning are facts about
    the artifact's bytes, and an environment variable that disagreed with them could only
    ever be wrong -- which is exactly what MILA_MODEL_FAMILY was, a second place to state
    something the record already knew. ModelWorker fills this in at startup; the protocol
    adapters read it per request.

    The default is Gemma because that is the default model, and it keeps every module
    importable (and the test suite runnable) without a store on the machine.
    """

    name: str = settings.model
    family: ModelFamily = ModelFamily.gemma
    variant: str = ""
    instruct: bool = True

    # Lineage. Carried because a license that requires attribution requires it of whoever
    # presents the model, and serving it over an API is presenting it.
    base_model: str = ""
    license: str = ""

    @property
    def attribution(self) -> str:
        return required_attribution_for(self.license)


loaded = LoadedModel()
