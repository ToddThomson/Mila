"""
Pydantic schemas for the OpenAI-compatible /v1/models endpoint.
"""
from pydantic import BaseModel


class ModelCapabilities(BaseModel):
    # Core modality
    completion: bool | None = None
    chat_completion: bool | None = None
    embeddings: bool | None = None

    # Audio / vision
    input_audio: bool | None = None
    output_audio: bool | None = None
    input_image: bool | None = None
    input_video: bool | None = None
    vision: bool | None = None
    message_file_upload: bool | None = None

    # Tool / interaction
    function_calling: bool | None = None
    tool_use: bool | None = None
    parallel_function_calls: bool | None = None
    json_output_mode: bool | None = None
    reasoning: bool | None = None
    logprobs: bool | None = None

    # Serving
    streaming: bool | None = None
    batch_jobs: bool | None = None
    fine_tuning: bool | None = None

    # Codex-CLI extensions (non-OpenAI)
    apply_patch: bool | None = None
    exec_command: bool | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    description: str | None = None
    display_name: str | None = None
    supports_streaming: bool | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_parallel_tool_calls: bool | None = None
    reasoning_summary_format: str | None = None
    slug: str | None = None
    capabilities: ModelCapabilities | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
