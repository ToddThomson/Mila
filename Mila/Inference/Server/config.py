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
    context_length: int = Field(8192, description="Maximum sequence length passed to fromPretrained().")
    device_index: int = Field(0, description="CUDA device ordinal.")
    strict_load: bool = Field(True, description="Raise on unrecognised parameter names during load.")

    # Generation defaults
    default_max_new_tokens: int = Field(256)
    default_temperature: float = Field(1.0)
    default_top_k: int = Field(0)

    # Server
    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    log_level: str = Field("info")

    # Protocol
    protocol: ProtocolMode = Field(ProtocolMode.openai, description="API protocol to expose.")


settings = Settings()