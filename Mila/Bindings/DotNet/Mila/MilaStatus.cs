namespace Mila;

/// <summary>Why a call into Mila failed.</summary>
public enum MilaStatus
{
    /// <summary>The call succeeded.</summary>
    Ok = 0,

    /// <summary>A null or invalid argument. Raised as <see cref="ArgumentException"/>, never as <see cref="MilaException"/>.</summary>
    InvalidArgument = 1,

    /// <summary>No installed model has that name.</summary>
    NotFound = 2,

    /// <summary>The model does not fit the device, or this build cannot load it.</summary>
    Refused = 3,

    /// <summary>This build of the native library was compiled without the function called.</summary>
    Unsupported = 4,

    /// <summary>Anything else the library reported.</summary>
    Runtime = 5,

    /// <summary>The native library and this assembly were built for different versions of the interface between them.</summary>
    Abi = 6,
}
