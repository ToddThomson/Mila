namespace Mila.Tests;

/// <summary>Section 3.3: mila_store_list_json, each record's fields crossing as snake_case JSON.</summary>
public sealed class StoreTests
{
    [Fact]
    public void ListHoldsTheMockStore()
    {
        IReadOnlyList<StoredModel> models = ModelStore.List();

        string[] expected = ["mock-gemma-it", "mock-llama-it", "mock-incomplete"];
        string[] names = models.Select(model => model.Name).ToArray();
        Assert.Equal(expected, names);
    }

    [Fact]
    public void EveryFieldOfARecordCrosses()
    {
        StoredModel gemma = ModelStore.List()[0];

        Assert.Equal("gemma", gemma.Architecture);
        Assert.Equal("gemma", gemma.Family);
        Assert.Equal("it", gemma.Variant);
        Assert.Equal("fp4", gemma.WeightQuantization);
        Assert.True(gemma.Instruct);
        Assert.True(gemma.Thinking);
        Assert.True(gemma.Tools);
        Assert.False(gemma.Agentic);
        Assert.False(gemma.Mtp);
        string[] modalities = ["text", "image"];
        Assert.Equal(modalities, gemma.Modalities);
        Assert.Equal(string.Empty, gemma.BaseModel);
        Assert.Equal("apache-2.0", gemma.License);
        Assert.Equal(string.Empty, gemma.Hub);
        Assert.Equal("2026-10-21T00:00:00Z", gemma.InstalledAt);
        Assert.Equal("/mock/store/mock-gemma-it/weights.safetensors", gemma.WeightsPath);
        Assert.Equal("/mock/store/mock-gemma-it/tokenizer.json", gemma.TokenizerPath);
        Assert.True(gemma.Complete);
        Assert.Equal(7340032000UL, gemma.BytesOnDisk);
    }

    [Fact]
    public void AnIncompleteModelIsListed()
    {
        IReadOnlyList<StoredModel> models = ModelStore.List();
        StoredModel incomplete = models.Single(model => model.Name == "mock-incomplete");

        Assert.False(incomplete.Complete);
        Assert.Equal("qwen", incomplete.Family);
    }
}
