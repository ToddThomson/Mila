namespace Mila.Tests;

/// <summary>Section 3.1's conventions as the C# library meets them: struct layouts, MILA_ERROR_ABI, and the status map.</summary>
public sealed unsafe class AbiTests
{
    [Fact]
    public void OptionsStructsHaveTheirCSizes()
    {
        Assert.Equal(32, sizeof(NativeLoadOptions));
        Assert.Equal(48, sizeof(NativeGenerateOptions));
    }

    [Fact]
    public void AnUnknownLoadOptionsSizeIsAnAbiError()
    {
        NativeLoadOptions options = new NativeLoadOptions();
        options.StructSize = 24;

        int status = NativeMethods.mila_model_from_store("mock-llama-it", in options, out ModelHandle model);
        string message = NativeResult.LastErrorMessage();
        model.Dispose();

        Assert.Equal((int)MilaStatus.Abi, status);
        Assert.Contains("struct_size is 24", message);
        Assert.True(model.IsInvalid);
    }

    [Fact]
    public void AnUnknownGenerateOptionsSizeIsAnAbiError()
    {
        using Model model = Model.FromStore("mock-llama-it");
        GenerateOptions defaults = new GenerateOptions();
        NativeGenerateOptions options = NativeGenerateOptions.From(defaults);
        options.StructSize = 40;
        int[] prompt = [10];
        int status;
        string message;

        int createStatus = NativeMethods.mila_stop_source_create(out StopSourceHandle stop);

        using (stop)
        {
            delegate* unmanaged<int, nint, void> onToken = &Ignore;

            fixed (int* tokens = prompt)
            {
                status = NativeMethods.mila_model_generate(model.Handle, tokens, prompt.Length, in options, onToken, nint.Zero, stop, out int reason);
            }

            message = NativeResult.LastErrorMessage();
        }

        Assert.Equal((int)MilaStatus.Ok, createStatus);
        Assert.Equal((int)MilaStatus.Abi, status);
        Assert.Contains("struct_size is 40", message);
    }

    [Fact]
    public void AnUnknownMaskBitIsAnArgumentError()
    {
        using Model model = Model.FromStore("mock-llama-it");
        GenerateOptions defaults = new GenerateOptions();
        NativeGenerateOptions options = NativeGenerateOptions.From(defaults);
        options.Set = 64;
        int[] prompt = [10];
        delegate* unmanaged<int, nint, void> onToken = &Ignore;
        using StopSourceHandle noStop = new StopSourceHandle();
        int status;

        fixed (int* tokens = prompt)
        {
            status = NativeMethods.mila_model_generate(model.Handle, tokens, prompt.Length, in options, onToken, nint.Zero, noStop, out int reason);
        }

        Assert.Equal((int)MilaStatus.InvalidArgument, status);
    }

    [Fact]
    public void AKnobLeftNullSetsNoMaskBit()
    {
        GenerateOptions options = new GenerateOptions { TopK = 0, Seed = 7 };

        NativeGenerateOptions native = NativeGenerateOptions.From(options);

        Assert.Equal(NativeGenerateOptions.SetTopK | NativeGenerateOptions.SetSeed, native.Set);
        Assert.Equal(64, native.MaxNewTokens);
    }

    [Theory]
    [InlineData(MilaStatus.NotFound)]
    [InlineData(MilaStatus.Refused)]
    [InlineData(MilaStatus.Unsupported)]
    [InlineData(MilaStatus.Runtime)]
    [InlineData(MilaStatus.Abi)]
    public void EveryFailureButAnInvalidArgumentIsAMilaException(MilaStatus status)
    {
        Exception exception = NativeResult.CreateException((int)status, "words");

        MilaException mila = Assert.IsType<MilaException>(exception);
        Assert.Equal(status, mila.Status);
        Assert.Equal("words", mila.Message);
    }

    [Fact]
    public void AnInvalidArgumentIsAnArgumentException()
    {
        Exception exception = NativeResult.CreateException((int)MilaStatus.InvalidArgument, "words");

        Assert.IsType<ArgumentException>(exception);
    }

    [Fact]
    public void ANullHandleIsAnInvalidArgument()
    {
        int status = NativeMethods.mila_model_context_length(new ModelHandle(), out long contextLength);

        Assert.Equal((int)MilaStatus.InvalidArgument, status);
    }

    [System.Runtime.InteropServices.UnmanagedCallersOnly]
    private static void Ignore(int token, nint userData)
    {
    }
}
