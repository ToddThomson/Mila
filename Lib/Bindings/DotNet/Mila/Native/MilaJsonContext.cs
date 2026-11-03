using System.Text.Json.Serialization;

namespace Mila;

/// <summary>The records that cross the ABI as JSON, serialized without reflection.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    RespectNullableAnnotations = true,
    RespectRequiredConstructorParameters = true)]
[JsonSerializable(typeof(IReadOnlyList<CudaDevice>))]
[JsonSerializable(typeof(IReadOnlyList<StoredModel>))]
[JsonSerializable(typeof(IReadOnlyList<ChatTurn>))]
[JsonSerializable(typeof(ParsedReply))]
internal sealed partial class MilaJsonContext : JsonSerializerContext
{
}
