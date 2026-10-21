namespace Mila;

/// <summary>One installed model.</summary>
/// <param name="Name">The name the model is opened by.</param>
/// <param name="Architecture">The architecture its manifest declares.</param>
/// <param name="Family">The family Mila loads it as -- "gemma", "llama" or "qwen" -- or empty when this build has none.</param>
/// <param name="Variant">The variant, such as "it".</param>
/// <param name="WeightQuantization">The format its weights are stored in.</param>
/// <param name="Instruct">Instruction-tuned, so it has a chat template.</param>
/// <param name="Thinking">It has a reasoning channel.</param>
/// <param name="Tools">It was trained for tool calls.</param>
/// <param name="Agentic">Mila's measured verdict that it works as an agent.</param>
/// <param name="Mtp">It carries a draft model this build decodes with.</param>
/// <param name="Modalities">The inputs this build runs for it.</param>
/// <param name="BaseModel">The model it was derived from.</param>
/// <param name="License">Its licence.</param>
/// <param name="Hub">Where it was installed from; empty when it was published from this machine.</param>
/// <param name="Owner">The owner on the hub.</param>
/// <param name="Repository">The repository on the hub.</param>
/// <param name="Revision">The revision installed.</param>
/// <param name="InstalledAt">When it was installed.</param>
/// <param name="WeightsPath">Its weights on disk.</param>
/// <param name="TokenizerPath">Its tokenizer on disk.</param>
/// <param name="Complete">False when a declared file is missing, which makes the model unloadable.</param>
/// <param name="BytesOnDisk">The space it takes.</param>
public sealed record StoredModel(
    string Name,
    string Architecture,
    string Family,
    string Variant,
    string WeightQuantization,
    bool Instruct,
    bool Thinking,
    bool Tools,
    bool Agentic,
    bool Mtp,
    IReadOnlyList<string> Modalities,
    string BaseModel,
    string License,
    string Hub,
    string Owner,
    string Repository,
    string Revision,
    string InstalledAt,
    string WeightsPath,
    string TokenizerPath,
    bool Complete,
    ulong BytesOnDisk);
