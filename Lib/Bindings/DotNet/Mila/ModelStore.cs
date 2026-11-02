using System.Text.Json;

namespace Mila;

/// <summary>The models installed on this machine.</summary>
public static class ModelStore
{
    /// <summary>Every installed model, complete or not.</summary>
    public static IReadOnlyList<StoredModel> List()
    {
        AbiVersion.EnsureCompatible();
        int status = NativeMethods.mila_store_list_json(out nint json);
        NativeResult.ThrowIfFailed(status);
        string text = NativeResult.TakeString(json);
        IReadOnlyList<StoredModel>? models = JsonSerializer.Deserialize(text, MilaJsonContext.Default.IReadOnlyListStoredModel);

        return models ?? throw new MilaException(MilaStatus.Runtime, "The store listed no models, not even an empty list.");
    }
}
