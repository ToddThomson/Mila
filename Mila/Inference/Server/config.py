from enum import Enum
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

    # Paths
    model_path: str = Field(..., description="Path to the Mila pretrained artifact.")
    tokenizer_path: str = Field(..., description="Path to the Mila Llama 3.2 tokenizer binary.")

    # Model
    model_name: str = Field("llama-3.2-3b-instruct", description="Model identifier returned in API responses.")
    context_length: int = Field(5120, description="Maximum sequence length passed to fromPretrained().")
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