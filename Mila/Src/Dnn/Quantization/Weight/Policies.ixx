// Mila - Dnn/Quantization/WeightQuant/Policies.ixx
// Weight quantization policy structs and concept.
//
// Consumers: Linear component and CudaLinearOp only.
// Neither KvCache nor any other subsystem imports this header.

module;
#include <concepts>

export module Dnn.Quantization.Weight.Policies;

import Dnn.TensorDataType;

namespace Mila::Dnn::Quant::Weight
{
    // -------------------------------------------------------------------------
    // NoWeightQuant
    //
    // Identity policy -- no quantization. Zero storage overhead, zero runtime
    // cost. All if constexpr branches on kIsQuantized compile away entirely.
    // Default for all Linear instantiations.
    // -------------------------------------------------------------------------
    export struct NoWeightQuant
    {
        static constexpr bool kIsQuantized = false;
        // FP32 is a harmless sentinel -- these fields are never read by Linear
        // on the unquantized path; all consumers guard with if constexpr (kIsQuantized).
        static constexpr TensorDataType kStorageDtype = TensorDataType::FP32;
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP32;
        static constexpr bool kPerChannel = false;
        static constexpr int kStorageBitsPerElement = 32;
    };

    // -------------------------------------------------------------------------
    // PerChannelFp8<TStorage>
    //
    // Per-output-channel FP8 weight quantization. Weights are quantized once at
    // load time (BF16 -> FP8_E4M3). One float32 scale per output channel,
    // computed as max(abs(W[o,:])) / 448.0f. Scales are uploaded to device and
    // held for the lifetime of the model. cuBLASLt FP8 matmul consumes weights
    // and scales natively -- no dequantization on the forward hot path.
    //
    // TStorage defaults to FP8_E4M3 (higher mantissa precision; correct for
    // stored weights). FP8_E5M2 is reserved for gradients and is not a Mila
    // inference target.
    // -------------------------------------------------------------------------
    export template<TensorDataType TStorage = TensorDataType::FP8_E4M3>
        struct PerChannelFp8
    {
        static constexpr bool kIsQuantized = true;
        static constexpr TensorDataType kStorageDtype = TStorage;
        static constexpr TensorDataType kScaleDtype = TensorDataType::FP32;
        static constexpr bool kPerChannel = true;
        static constexpr int kStorageBitsPerElement = 8;
    };

    // -------------------------------------------------------------------------
    // PerGroupInt4<kGroupSize>
    //
    // Per-group INT4 weight quantization (W4A16). Weights are stored as packed
    // uint8: two INT4 nibbles per byte (low nibble = even column, high nibble =
    // odd column). One float32 scale and one INT4 zero point per group of
    // kGroupSize input channels. Supports symmetric quantization (zero_points =
    // nullptr, implicit zero = 8) and asymmetric (explicit INT4 zero points).
    //
    // kGroupSize defaults to 128, matching the most common GPTQ checkpoint format.
    // 64 is also supported by the cuda_w4a16_gemm kernel.
    //
    // kPerChannel = false distinguishes this policy from PerChannelFp8 at
    // compile time -- CudaLinearOp uses kPerChannel as a dispatch discriminator.
    // -------------------------------------------------------------------------
    export template<int kGroupSize = 128>
        struct PerGroupInt4
    {
        static constexpr bool            kIsQuantized           = true;
        static constexpr TensorDataType  kStorageDtype          = TensorDataType::UINT8;  // packed INT4 in uint8 bytes
        static constexpr TensorDataType  kScaleDtype            = TensorDataType::FP32;
        static constexpr bool            kPerChannel            = false;  // per-group, not per-channel
        static constexpr int             kQuantizationGroupSize = kGroupSize;
        static constexpr bool            kIsFp4E2M1             = false;
        static constexpr int             kStorageBitsPerElement = 4;      // two nibbles per byte
    };

    // -------------------------------------------------------------------------
    // PerGroupFp4<kGroupSize>
    //
    // Per-group FP4 E2M1 weight quantization (W4A16). Weights are stored as
    // packed uint8: two FP4_E2M1 nibbles per byte (low nibble = even column,
    // high nibble = odd column). One float32 scale per group of kGroupSize input
    // channels; scale[g] = max(|W[g,:]|) / 6.0f (6.0 = max E2M1 representable).
    //
    // Dequantization per element:
    //   nibble = low nibble if k even, high nibble if k odd
    //   W_f32  = fp4_e2m1_lut[nibble] * scale[k / kGroupSize]
    //
    // The E2M1 representable positive values (nibbles 0..7):
    //   {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    // Negative values (nibbles 8..15): sign-magnitude mirror.
    //
    // No zero-points -- sign is encoded directly in the nibble (bit 3).
    // Symmetric quantization only.
    //
    // kGroupSize defaults to 128. VRAM reduction vs BF16: 4x.
    // Native FP4 compute (Blackwell SM 10.0+) can consume this format directly
    // when available; the storage layout is forward-compatible.
    // -------------------------------------------------------------------------
    export template<int kGroupSize = 128>
        struct PerGroupFp4
    {
        static constexpr bool            kIsQuantized           = true;
        static constexpr TensorDataType  kStorageDtype          = TensorDataType::UINT8;  // packed FP4 E2M1: 2 nibbles per byte, physically uint8
        static constexpr TensorDataType  kScaleDtype            = TensorDataType::FP32;
        static constexpr bool            kPerChannel            = false;  // per-group, not per-channel
        static constexpr int             kQuantizationGroupSize = kGroupSize;
        static constexpr bool            kIsFp4E2M1             = true;
        static constexpr int             kStorageBitsPerElement = 4;      // two nibbles per byte
    };

    // -------------------------------------------------------------------------
    // WeightQuantPolicy concept
    //
    // Any type satisfying this concept may be used as the TWeightQuant parameter
    // on Linear and CudaLinearOp. The concept is intentionally narrow -- only the
    // fields that Linear needs to make compile-time decisions are required.
    //
    // NoWeightQuant satisfies this concept (kStorageDtype = FP32 sentinel is never
    // read; Linear guards all dtype-consuming paths with if constexpr (kIsQuantized)).
    // -------------------------------------------------------------------------
    export template<typename T>
        concept WeightQuantPolicy = requires
    {
        { T::kIsQuantized  } -> std::convertible_to<bool>;
        {
            T::kStorageDtype
        } -> std::convertible_to<TensorDataType>;
        {
            T::kScaleDtype
        } -> std::convertible_to<TensorDataType>;
        {
            T::kPerChannel
        } -> std::convertible_to<bool>;
        // Bits of the PRIMARY weight tensor per logical element -- what Linear needs to
        // size the packed allocation. It states the fact the allocation used to infer from
        // kPerChannel, which only ever meant "nibble-packed" because every per-group policy
        // happened to be 4-bit. A policy whose format spills into a companion plane counts
        // only the primary tensor here; the companion carries its own extent.
        {
            T::kStorageBitsPerElement
        } -> std::convertible_to<int>;
    };

    // -------------------------------------------------------------------------
    // PerGroupCodebook2<kGroupSize> / PerGroupCodebook3<kGroupSize>
    //
    // Sub-4-bit codebook weight quantization (W2A16 / W3A16). Codes index a per-tensor
    // codebook rather than a uniform step ladder, which is what lets the format absorb
    // asymmetry without a zero point -- measured as both cheaper and materially better
    // than a per-group zero (Specifications/Qwen3.8.md, Phase 0).
    //
    // Codes are produced OFFLINE by the converter with calibration and GPTQ error
    // compensation; loadParameter() uploads them and never quantizes. Data-free
    // round-to-nearest at these bit widths destroys the model outright, so unlike every
    // other policy here there is no quantize-on-load path.
    //
    //   PerGroupCodebook2: 2-bit codes, 4-entry table, FP16 scale per group of 32
    //                      -> 2 + 16/32 = 2.5 bits per weight
    //   PerGroupCodebook3: 3-bit codes, 8-entry table, FP16 scale per group of 64
    //                      -> 3 + 16/64 = 3.25 bits per weight
    //
    // The 3-bit format stores the low two bits of each code in the primary tensor and the
    // third in a separate byte-aligned plane, so both policies share one kernel family and
    // keep tile loads aligned. kStorageBitsPerElement describes the PRIMARY tensor only --
    // 2 for both -- and the plane carries its own extent.
    //
    // The normative packed layout and its CPU reference codec are in CodebookPacking.ixx.
    // -------------------------------------------------------------------------
    export template<int kGroupSize = 32>
        struct PerGroupCodebook2
    {
        static constexpr bool            kIsQuantized           = true;
        static constexpr TensorDataType  kStorageDtype          = TensorDataType::UINT8;
        static constexpr TensorDataType  kScaleDtype            = TensorDataType::FP16;
        static constexpr bool            kPerChannel            = false;
        static constexpr int             kQuantizationGroupSize = kGroupSize;
        static constexpr bool            kIsFp4E2M1             = false;
        static constexpr int             kStorageBitsPerElement = 2;
        static constexpr bool            kIsCodebook            = true;
        static constexpr int             kCodeBits              = 2;
        static constexpr int             kCodebookEntries       = 4;
        static constexpr bool            kHasHighBitPlane       = false;
    };

    export template<int kGroupSize = 64>
        struct PerGroupCodebook3
    {
        static constexpr bool            kIsQuantized           = true;
        static constexpr TensorDataType  kStorageDtype          = TensorDataType::UINT8;
        static constexpr TensorDataType  kScaleDtype            = TensorDataType::FP16;
        static constexpr bool            kPerChannel            = false;
        static constexpr int             kQuantizationGroupSize = kGroupSize;
        static constexpr bool            kIsFp4E2M1             = false;
        static constexpr int             kStorageBitsPerElement = 2;  // primary plane only
        static constexpr bool            kIsCodebook            = true;
        static constexpr int             kCodeBits              = 3;
        static constexpr int             kCodebookEntries       = 8;
        static constexpr bool            kHasHighBitPlane       = true;
    };

    // -------------------------------------------------------------------------
    // Companion-tensor traits
    //
    // A quantized format may need storage beyond the packed weight and its scales.
    // Linear serves those companions without naming a format: it asks the policy what it
    // declares. Detection is STRUCTURAL, so a policy defined outside this module -- a
    // research format under Src/Experimental, which core must never import -- satisfies
    // these without the core knowing the type exists.
    //
    // A policy that declares neither compiles to exactly what it compiled to before these
    // existed; every consumer guards with if constexpr.
    // -------------------------------------------------------------------------

    /// The format decodes through a per-tensor table, which travels as its own tensor.
    export template<typename T>
        concept HasCodebookTable = requires
    {
        { T::kIsCodebook } -> std::convertible_to<bool>;
        { T::kCodebookEntries } -> std::convertible_to<int>;
    } && T::kIsCodebook;

    /// The format spills one bit per element into a second byte-aligned plane, so the
    /// primary tensor's kStorageBitsPerElement does not account for the whole code.
    export template<typename T>
        concept HasHighBitPlane = requires
    {
        { T::kHasHighBitPlane } -> std::convertible_to<bool>;
    } && T::kHasHighBitPlane;

    // Verify all concrete policies satisfy the concept at definition time.
    static_assert(WeightQuantPolicy<NoWeightQuant>);
    static_assert(WeightQuantPolicy<PerChannelFp8<>>);
    static_assert(WeightQuantPolicy<PerGroupInt4<>>);
    static_assert(WeightQuantPolicy<PerGroupInt4<64>>);
    static_assert(WeightQuantPolicy<PerGroupFp4<>>);
    static_assert(WeightQuantPolicy<PerGroupFp4<64>>);
    static_assert(WeightQuantPolicy<PerGroupCodebook2<>>);
    static_assert(WeightQuantPolicy<PerGroupCodebook3<>>);
    static_assert(WeightQuantPolicy<PerGroupCodebook2<64>>);
    static_assert(WeightQuantPolicy<PerGroupCodebook3<128>>);

    // The companion traits must select exactly the policies that declare them: a codebook
    // policy allocating no table, or an FP4 policy allocating one, would both be silent.
    static_assert(HasCodebookTable<PerGroupCodebook2<>>);
    static_assert(HasCodebookTable<PerGroupCodebook3<>>);
    static_assert(!HasCodebookTable<PerGroupFp4<>>);
    static_assert(!HasCodebookTable<NoWeightQuant>);
    static_assert(HasHighBitPlane<PerGroupCodebook3<>>);
    static_assert(!HasHighBitPlane<PerGroupCodebook2<>>);
    static_assert(!HasHighBitPlane<PerGroupFp4<>>);

}
