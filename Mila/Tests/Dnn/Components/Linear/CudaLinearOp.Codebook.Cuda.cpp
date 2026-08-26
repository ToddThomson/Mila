/**
 * @file CudaLinearOp.Codebook.Cuda.cpp
 * @brief The Phase 1 dispatch gate: a codebook weight policy resolving through the
 * OperationTraits table to working decode and prefill forwards on real Mila tensors.
 *
 * CodebookGemv.Cuda.cpp already proves the kernels against the normative codec. What is
 * under test here is everything between: that the policies dispatch to the one Linear
 * operation every other weight format uses, that that operation can carry pre-quantized
 * weights it never quantizes and owns none of them, and that the numbers survive the trip
 * through Tensor and ExecutionContext unchanged.
 *
 * Compiled only under MILA_ENABLE_CUDA; each test skips if no device is present.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>

import Mila;
import Compute.ExecutionContext;
import Compute.OperationTraits;
import Serialization.Tensor;
import Dnn.Quantization.Weight.CodebookPacking;
import Dnn.Quantization.Weight.Policies;
import Compute.CudaLinearOp;


namespace Mila::Tests::Dnn::Quantization
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Compute::Cuda::Linear;
    using namespace Mila::Dnn::Quant::Weight;

    namespace
    {
        template<typename TPolicy>
        using CodebookOpFor = typename OperationTraits<
            OperationType::LinearOp, DeviceType::Cuda, TensorDataType::BF16, TPolicy>::type;

        // The dispatch claim, checked at compile time: the rows exist, and they name the
        // SAME CudaLinearOp every other weight policy resolves to -- the policy selects the
        // decode kernel and the prefill strategy, not a second operation. A missing row
        // would not reach these assertions at all: the tuple would name the undefined
        // primary and the alias above would fail to compile, which is the diagnostic
        // OperationTraits.Template.ixx promises.
        static_assert( OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
            TensorDataType::BF16, PerGroupCodebook2<32>> );
        static_assert( OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
            TensorDataType::BF16, PerGroupCodebook3<64>> );
        static_assert( std::is_same_v<CodebookOpFor<PerGroupCodebook2<32>>,
            CudaLinearOp<TensorDataType::BF16, PerGroupCodebook2<32>>> );
        static_assert( std::is_same_v<CodebookOpFor<PerGroupCodebook3<64>>,
            CudaLinearOp<TensorDataType::BF16, PerGroupCodebook3<64>>> );

        // The consolidation's load-bearing property: a member a format has no meaning for
        // is CONSTRAINED AWAY, not present and throwing. The two directions are asserted
        // differently because MSVC treats a failed constraint on a NON-TEMPLATE member as a
        // hard error rather than substitution failure, so the negative cannot be spelled as
        // !requires{ op.member(...) } -- it fails to compile instead of evaluating false.
        // The positive form works, and the discriminator that gates it is checked directly.
        static_assert( requires( CudaLinearOp<TensorDataType::BF16, PerGroupFp4<128>> op,
            const Mila::Dnn::Serialization::ITensorBlob& blob,
            ITensor& weight, ITensor& scales, const shape_t& shape )
        {
            op.quantize( blob, weight, scales, shape );
        } );
        static_assert( requires( CudaLinearOp<TensorDataType::BF16, PerGroupCodebook3<64>> op,
            ITensor* codebook, ITensor* plane )
        {
            op.setCodebookTensors( codebook, plane );
        } );

        static_assert( CudaLinearOp<TensorDataType::BF16, PerGroupCodebook2<32>>::kIsCodebookWeight );
        static_assert( !CudaLinearOp<TensorDataType::BF16, PerGroupFp4<128>>::kIsCodebookWeight );

        // Both formats reach the same striped two-phase prefill; only the cap differs, and
        // only the expansion kernel inside it differs.
        static_assert( CudaLinearOp<TensorDataType::BF16, PerGroupCodebook2<32>>::kUsesStagedPrefill );
        static_assert( CudaLinearOp<TensorDataType::BF16, PerChannelFp8<>>::kUsesStagedPrefill );

        // A precision the GEMV kernels cannot honor must have no row at all, per the
        // OperationTraits contract -- a missing specialization, never a poisoned one.
        static_assert( !OperationSupported<OperationType::LinearOp, DeviceType::Cuda,
            TensorDataType::FP32, PerGroupCodebook2<32>> );

        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;

        float roundThroughBf16( float value )
        {
            std::uint32_t bits = std::bit_cast<std::uint32_t>( value );
            bits += 0x7FFFu + ( ( bits >> 16 ) & 1u );

            return std::bit_cast<float>( bits & 0xFFFF0000u );
        }

        struct PackedTensor
        {
            std::vector<std::uint8_t> codes;
            std::vector<std::uint8_t> planeTwoBits;
            std::vector<std::uint8_t> planeOneBit;
            std::vector<std::uint16_t> scaleBits;
            std::vector<float> codebook;
            std::vector<float> activations;
        };

        PackedTensor makePackedTensor(
            dim_t rows, dim_t columns, dim_t groupSize, int entries, unsigned seed )
        {
            PackedTensor packed;
            std::mt19937 generator( seed );
            std::uniform_int_distribution<int> codeDistribution( 0, entries - 1 );
            std::uniform_real_distribution<float> scaleDistribution( 0.01f, 2.0f );
            std::normal_distribution<float> activationDistribution( 0.0f, 1.0f );

            packed.codes.resize( static_cast<std::size_t>( rows * columns ) );

            for ( auto& code : packed.codes )
                code = static_cast<std::uint8_t>( codeDistribution( generator ) );

            packed.scaleBits.resize(
                static_cast<std::size_t>( rows * groupsPerRow( columns, groupSize ) ) );

            for ( auto& scale : packed.scaleBits )
                scale = floatToHalfBits( scaleDistribution( generator ) );

            packed.codebook.resize( static_cast<std::size_t>( entries ) );

            for ( auto& entry : packed.codebook )
                entry = std::uniform_real_distribution<float>( -1.0f, 1.0f )( generator );

            packed.activations.resize( static_cast<std::size_t>( columns ) );

            for ( auto& activation : packed.activations )
                activation = activationDistribution( generator );

            packed.planeTwoBits.resize(
                static_cast<std::size_t>( rows * packedRowBytesForTwoBitCodes( columns ) ) );

            if ( entries == 4 ) {
                packTwoBitCodes( packed.codes.data(), rows, columns, packed.planeTwoBits.data() );
            } else {
                packed.planeOneBit.resize(
                    static_cast<std::size_t>( rows * packedRowBytesForOneBitPlane( columns ) ) );
                packThreeBitCodes( packed.codes.data(), rows, columns,
                    packed.planeTwoBits.data(), packed.planeOneBit.data() );
            }

            return packed;
        }

        std::vector<float> referenceRowSums(
            const PackedTensor& packed, dim_t rows, dim_t columns, dim_t groupSize )
        {
            std::vector<float> dequantized( static_cast<std::size_t>( rows * columns ) );
            dequantizeCodes( packed.codes.data(), packed.scaleBits.data(), packed.codebook.data(),
                rows, columns, groupSize, dequantized.data() );

            std::vector<float> expected( static_cast<std::size_t>( rows ) );

            for ( dim_t row = 0; row < rows; ++row ) {
                double accumulator = 0.0;

                for ( dim_t column = 0; column < columns; ++column ) {
                    const float activation = roundThroughBf16(
                        packed.activations[static_cast<std::size_t>( column )] );
                    accumulator += static_cast<double>( activation )
                        * dequantized[static_cast<std::size_t>( row * columns + column )];
                }

                expected[static_cast<std::size_t>( row )] = static_cast<float>( accumulator );
            }

            return expected;
        }

        std::unique_ptr<IExecutionContext> makeContextOrSkip()
        {
            try {
                return createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& ) {
                return nullptr;
            }
        }

        // The four tensors Linear owns for a codebook policy, held together so a test can
        // bind them the way onBuilding() does. Raw cudaMemcpy rather than copy(): these
        // carry packed bytes and IEEE half bits, neither of which has a host tensor type
        // that would round-trip them unchanged.
        struct CodebookWeightTensors
        {
            Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource> codes;
            Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource> highPlane;
            Tensor<TensorDataType::FP16, CudaDeviceMemoryResource> scales;
            Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> codebook;
            bool hasHighPlane{ false };
        };

        template<typename TPolicy>
        CodebookWeightTensors uploadWeights(
            const PackedTensor& packed, dim_t rows, dim_t columns )
        {
            constexpr dim_t kGroupSize = TPolicy::kQuantizationGroupSize;
            const dim_t groups = groupsPerRow( columns, kGroupSize );

            using CodeTensor = Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource>;
            using ScaleTensor = Tensor<TensorDataType::FP16, CudaDeviceMemoryResource>;
            using TableTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;

            CodebookWeightTensors tensors{
                CodeTensor( Device::Cuda( 0 ), shape_t{ rows, columns / 4 } ),
                CodeTensor( Device::Cuda( 0 ),
                    shape_t{ rows, TPolicy::kHasHighBitPlane ? columns / 8 : 1 } ),
                ScaleTensor( Device::Cuda( 0 ), shape_t{ rows, groups } ),
                TableTensor( Device::Cuda( 0 ), shape_t{ TPolicy::kCodebookEntries } ),
                TPolicy::kHasHighBitPlane };

            cudaMemcpy( tensors.codes.rawData(), packed.planeTwoBits.data(),
                packed.planeTwoBits.size(), cudaMemcpyHostToDevice );
            cudaMemcpy( tensors.scales.rawData(), packed.scaleBits.data(),
                packed.scaleBits.size() * sizeof( std::uint16_t ), cudaMemcpyHostToDevice );
            cudaMemcpy( tensors.codebook.rawData(), packed.codebook.data(),
                packed.codebook.size() * sizeof( float ), cudaMemcpyHostToDevice );

            if constexpr ( TPolicy::kHasHighBitPlane )
            {
                cudaMemcpy( tensors.highPlane.rawData(), packed.planeOneBit.data(),
                    packed.planeOneBit.size(), cudaMemcpyHostToDevice );
            }

            return tensors;
        }

        template<typename TPolicy, typename TOp>
        void bindWeights( TOp& op, CodebookWeightTensors& tensors )
        {
            op.setParameters( &tensors.codes, nullptr );
            op.setWeightScales( &tensors.scales );
            op.setCodebookTensors( &tensors.codebook,
                tensors.hasHighPlane ? &tensors.highPlane : nullptr );
        }

        template<typename TPolicy>
        void runDecodeAndCompare( dim_t rows, dim_t columns, int entries, unsigned seed )
        {
            constexpr dim_t kGroupSize = TPolicy::kQuantizationGroupSize;

            auto context = makeContextOrSkip();

            if ( context == nullptr )
                GTEST_SKIP() << "CUDA device not available";

            const PackedTensor packed =
                makePackedTensor( rows, columns, kGroupSize, entries, seed );
            const std::vector<float> expected =
                referenceRowSums( packed, rows, columns, kGroupSize );

            LinearConfig config( columns, rows );
            config.withBias( false );

            CodebookOpFor<TPolicy> op( context.get(), config );
            auto tensors = uploadWeights<TPolicy>( packed, rows, columns );
            bindWeights<TPolicy>( op, tensors );
            op.build( BuildContext( shape_t{ 1, columns }, RuntimeMode::Inference, false ) );

            ASSERT_TRUE( op.isBuilt() );

            // The operation owns NOTHING: the component owns every packed buffer and the
            // operation caches borrowed pointers. Those bytes are the component's
            // parameter memory, never the operation's state, so a non-zero figure here
            // would mean the op had started allocating behind the component's back and
            // the footprint predictor would be double-counting.
            EXPECT_EQ( op.getStateMemorySize(), 0u );

            HostFp32 host_input( Device::Cpu(), shape_t{ 1, columns } );

            for ( dim_t column = 0; column < columns; ++column )
                host_input.data()[column] = packed.activations[static_cast<std::size_t>( column )];

            DeviceBf16 device_input( Device::Cuda( 0 ), shape_t{ 1, columns } );
            DeviceBf16 device_output( Device::Cuda( 0 ), shape_t{ 1, rows } );
            copy( host_input, device_input, context.get() );
            context->synchronize();

            op.forward( device_input, device_output );
            context->synchronize();

            auto host_output = toHost<TensorDataType::FP32>( device_output, context.get() );
            context->synchronize();

            for ( dim_t row = 0; row < rows; ++row ) {
                const float reference = expected[static_cast<std::size_t>( row )];
                // The BF16 output store dominates at 2^-8 relative; the absolute floor
                // covers accumulation order for rows that sum near zero.
                const float tolerance = 0.01f + 0.005f * std::abs( reference );
                EXPECT_NEAR( host_output.data()[row], reference, tolerance ) << "row " << row;
            }
        }
    }

    TEST( CodebookLinearOpCuda, TwoBitGroup32DecodeMatchesCodec )
    {
        runDecodeAndCompare<PerGroupCodebook2<32>>( 64, 512, 4, 20260823u );
    }

    TEST( CodebookLinearOpCuda, ThreeBitGroup64DecodeMatchesCodec )
    {
        runDecodeAndCompare<PerGroupCodebook3<64>>( 48, 3072, 8, 20260824u );
    }

    namespace
    {
        // Both legs are measured against the double-precision codec, not against each
        // other. Decode and prefill share no kernel, no staging and no library call, so a
        // structural error in either -- stride, transpose, scale indexing -- moves that
        // leg away from the reference by O(sum of |terms|) and is caught here.
        //
        // There is deliberately NO magnitude sweep, unlike Linear's
        // Forward_Fp4PrefillMatchesDecodeAcrossTokenMagnitudes. That sweep exists because
        // the FP8 path quantizes activations, so a per-tensor scale crushes small tokens
        // and the failure is invisible at one magnitude. This path never touches
        // activations: both legs read the same BF16 input. Scaling a row by 1e7 therefore
        // tests nothing here and only destroys conditioning -- measured, it produced four
        // failures per row on the four output rows whose dot product cancels to near zero,
        // which is a property of the fixture rather than of the kernels.
        //
        // The budget is the standard backward-error form for a dot product: proportional
        // to the sum of absolute products, not to the result. A result that cancels to
        // near zero has no relative accuracy to demand, and demanding it is what made the
        // first version of this test fail against correct kernels.
        template<typename TPolicy>
        void runPrefillAndDecodeAgainstCodec(
            dim_t rows, dim_t columns, int entries, unsigned seed )
        {
            constexpr dim_t kGroupSize = TPolicy::kQuantizationGroupSize;
            constexpr dim_t kBatchRows = 16;

            auto context = makeContextOrSkip();

            if ( context == nullptr )
                GTEST_SKIP() << "CUDA device not available";

            const PackedTensor packed =
                makePackedTensor( rows, columns, kGroupSize, entries, seed );

            std::vector<float> dequantized( static_cast<std::size_t>( rows * columns ) );
            dequantizeCodes( packed.codes.data(), packed.scaleBits.data(), packed.codebook.data(),
                rows, columns, kGroupSize, dequantized.data() );

            // Every batch row is an independent draw: 16 scaled copies of one vector would
            // make every row the same test.
            std::mt19937 generator( seed ^ 0x5eedu );
            std::normal_distribution<float> activationDistribution( 0.0f, 1.0f );

            HostFp32 host_input( Device::Cpu(), shape_t{ kBatchRows, columns } );

            for ( dim_t m = 0; m < kBatchRows; ++m )
                for ( dim_t k = 0; k < columns; ++k )
                    host_input.data()[m * columns + k] = activationDistribution( generator );

            // The two legs hold the weight at different precisions, so they get different
            // references. Decode multiplies codebook[code] * scale in an FP32 register and
            // never stores it; the two-phase prefill materializes that product in a BF16
            // staging buffer, costing 8 mantissa bits per weight before the GEMM sees it.
            //
            // Measured, this is not a rounding quibble: against a single FP32-weight
            // reference, decode matched at every one of 4096 comparisons while prefill
            // missed 351 -- all of them outputs whose dot product cancels toward zero,
            // where a 2^-9 relative error per weight is the whole answer. Holding both
            // legs to one reference would have meant either declaring a correct kernel
            // broken or loosening the budget until it stopped testing anything.
            //
            // The property is inherent to staging, not to this implementation: the
            // production FP4 two-phase path has it as well. It matters for Section 5
            // because prefill and decode over the same weights are then not bit-comparable
            // at the model level -- a prompt reprocessed token-by-token will not reproduce
            // its batched logits exactly.
            std::vector<float> dequantizedThroughBf16( dequantized.size() );

            for ( std::size_t index = 0; index < dequantized.size(); ++index )
                dequantizedThroughBf16[index] = roundThroughBf16( dequantized[index] );

            std::vector<double> expectedDecode( static_cast<std::size_t>( kBatchRows * rows ) );
            std::vector<double> expectedPrefill( static_cast<std::size_t>( kBatchRows * rows ) );
            std::vector<double> budget( static_cast<std::size_t>( kBatchRows * rows ) );

            for ( dim_t m = 0; m < kBatchRows; ++m ) {
                for ( dim_t row = 0; row < rows; ++row ) {
                    double decodeAccumulator = 0.0;
                    double prefillAccumulator = 0.0;
                    double magnitude = 0.0;

                    for ( dim_t column = 0; column < columns; ++column ) {
                        const std::size_t weightIndex =
                            static_cast<std::size_t>( row * columns + column );
                        const double activation = static_cast<double>(
                            roundThroughBf16( host_input.data()[m * columns + column] ) );

                        const double term = activation * dequantized[weightIndex];
                        decodeAccumulator += term;
                        prefillAccumulator += activation * dequantizedThroughBf16[weightIndex];
                        magnitude += std::abs( term );
                    }

                    const std::size_t index = static_cast<std::size_t>( m * rows + row );
                    expectedDecode[index] = decodeAccumulator;
                    expectedPrefill[index] = prefillAccumulator;
                    // Accumulation, plus the BF16 output store. The first coefficient is
                    // MEASURED, not derived: at C = 3072 the staged cuBLASLt GEMM's worst
                    // observed error was 1.8e-4 of the magnitude, well above the
                    // sqrt(C)*eps bound of ~3e-6 the decode GEMV sits under. 5e-4 gives
                    // that ~2.8x of headroom while staying ~2000x tighter than a
                    // structural error, which costs O(1) of the magnitude and cannot hide
                    // under any coefficient in this range.
                    budget[index] = 5e-4 * magnitude + 5e-3 * std::abs( decodeAccumulator );
                }
            }

            LinearConfig config( columns, rows );
            config.withBias( false );

            CodebookOpFor<TPolicy> op( context.get(), config );
            auto tensors = uploadWeights<TPolicy>( packed, rows, columns );
            bindWeights<TPolicy>( op, tensors );
            op.build( BuildContext( shape_t{ kBatchRows, columns }, RuntimeMode::Inference, false ) );

            DeviceBf16 batch_input( Device::Cuda( 0 ), shape_t{ kBatchRows, columns } );
            DeviceBf16 batch_output( Device::Cuda( 0 ), shape_t{ kBatchRows, rows } );
            copy( host_input, batch_input, context.get() );
            context->synchronize();

            op.forward( batch_input, batch_output );
            context->synchronize();

            auto prefill_host = toHost<TensorDataType::FP32>( batch_output, context.get() );
            context->synchronize();

            std::vector<float> prefill(
                prefill_host.data(),
                prefill_host.data() + static_cast<std::size_t>( kBatchRows * rows ) );

            for ( dim_t m = 0; m < kBatchRows; ++m )
                for ( dim_t row = 0; row < rows; ++row ) {
                    const std::size_t index = static_cast<std::size_t>( m * rows + row );
                    EXPECT_NEAR( prefill[index], expectedPrefill[index], budget[index] )
                        << "prefill, batch row " << m << ", output " << row;
                }

            // Decode bypasses cuBLASLt entirely, so the same instance serves both legs
            // regardless of the shape it was built for.
            for ( dim_t m = 0; m < kBatchRows; ++m ) {
                HostFp32 host_row( Device::Cpu(), shape_t{ 1, columns } );

                for ( dim_t k = 0; k < columns; ++k )
                    host_row.data()[k] = host_input.data()[m * columns + k];

                DeviceBf16 row_input( Device::Cuda( 0 ), shape_t{ 1, columns } );
                DeviceBf16 row_output( Device::Cuda( 0 ), shape_t{ 1, rows } );
                copy( host_row, row_input, context.get() );
                context->synchronize();

                op.forward( row_input, row_output );
                context->synchronize();

                auto decode_host = toHost<TensorDataType::FP32>( row_output, context.get() );
                context->synchronize();

                for ( dim_t row = 0; row < rows; ++row ) {
                    const std::size_t index = static_cast<std::size_t>( m * rows + row );
                    EXPECT_NEAR( decode_host.data()[row], expectedDecode[index], budget[index] )
                        << "decode, batch row " << m << ", output " << row;
                }
            }
        }
    }

    namespace
    {
        // Prefill walks the output channels in strips so the BF16 staging buffer stays under
        // a cap instead of growing with the weight matrix (Qwen3.8.md sections 5 and 8).
        // Every real shape in this file is far below the shipped 32 MiB cap, so without a
        // constructed cap the striped path is dead code in the suite.
        //
        // The claim under test is the one the design rests on: striping changes only WHERE
        // the dequantized weights live, never the arithmetic, because each output is a dot
        // product over exactly one strip and is never summed across strips. So the check is
        // not a tolerance -- it is BITWISE equality against the same operation run in one
        // pass. A tolerance here would pass just as happily on a version that split the
        // contraction instead, which would round partial sums to BF16 and is precisely the
        // mistake this shape of striping avoids.
        template<typename TPolicy>
        void runStripedPrefillMatchesUnstriped(
            dim_t rows, dim_t columns, int entries, unsigned seed,
            std::size_t stagingCap, dim_t expectedStripRows, dim_t expectedTrailingRows )
        {
            constexpr dim_t kGroupSize = TPolicy::kQuantizationGroupSize;
            constexpr dim_t kBatchRows = 16;

            auto context = makeContextOrSkip();

            if ( context == nullptr )
                GTEST_SKIP() << "CUDA device not available";

            const PackedTensor packed =
                makePackedTensor( rows, columns, kGroupSize, entries, seed );

            std::mt19937 generator( seed ^ 0x5721u );
            std::normal_distribution<float> activationDistribution( 0.0f, 1.0f );

            HostFp32 host_input( Device::Cpu(), shape_t{ kBatchRows, columns } );

            for ( dim_t m = 0; m < kBatchRows; ++m )
                for ( dim_t k = 0; k < columns; ++k )
                    host_input.data()[m * columns + k] = activationDistribution( generator );

            DeviceBf16 batch_input( Device::Cuda( 0 ), shape_t{ kBatchRows, columns } );
            copy( host_input, batch_input, context.get() );
            context->synchronize();

            const auto runWithCap = [&]( std::size_t cap, dim_t* stripRows, dim_t* trailingRows )
            {
                LinearConfig config( columns, rows );
                config.withBias( false );

                CodebookOpFor<TPolicy> op( context.get(), config, cap );
                auto tensors = uploadWeights<TPolicy>( packed, rows, columns );
                bindWeights<TPolicy>( op, tensors );
                op.build( BuildContext( shape_t{ kBatchRows, columns }, RuntimeMode::Inference, false ) );

                *stripRows = op.getStripRows();
                *trailingRows = op.getTrailingStripRows();

                DeviceBf16 batch_output( Device::Cuda( 0 ), shape_t{ kBatchRows, rows } );
                op.forward( batch_input, batch_output );
                context->synchronize();

                auto host = toHost<TensorDataType::FP32>( batch_output, context.get() );
                context->synchronize();

                return std::vector<float>( host.data(),
                    host.data() + static_cast<std::size_t>( kBatchRows * rows ) );
            };

            dim_t wholeStrip = 0, wholeTrailing = 0;
            const std::vector<float> unstriped =
                runWithCap( std::size_t( 1 ) << 40, &wholeStrip, &wholeTrailing );

            EXPECT_EQ( wholeStrip, rows ) << "a cap this large must not stripe at all";
            EXPECT_EQ( wholeTrailing, rows );

            dim_t stripRows = 0, trailingRows = 0;
            const std::vector<float> striped = runWithCap( stagingCap, &stripRows, &trailingRows );

            EXPECT_EQ( stripRows, expectedStripRows );
            EXPECT_EQ( trailingRows, expectedTrailingRows );
            ASSERT_GT( rows, stripRows ) << "the cap must actually force more than one strip";

            ASSERT_EQ( unstriped.size(), striped.size() );

            for ( dim_t m = 0; m < kBatchRows; ++m )
                for ( dim_t row = 0; row < rows; ++row )
                {
                    const std::size_t index = static_cast<std::size_t>( m * rows + row );
                    EXPECT_EQ( std::bit_cast<std::uint32_t>( striped[index] ),
                        std::bit_cast<std::uint32_t>( unstriped[index] ) )
                        << "batch row " << m << ", output " << row
                        << " (strip " << ( row / stripRows ) << ")";
                }
        }
    }

    // 256 rows x 512 columns is 256 KiB staged whole; a 70 KiB cap asks for four strips and
    // divides evenly, so every strip is the same width and none is trailing.
    TEST( CodebookLinearOpCuda, TwoBitStripedPrefillMatchesUnstriped )
    {
        runStripedPrefillMatchesUnstriped<PerGroupCodebook2<32>>(
            256, 512, 4, 20260901u, 70u * 1024, 64, 64 );
    }

    TEST( CodebookLinearOpCuda, ThreeBitStripedPrefillMatchesUnstriped )
    {
        runStripedPrefillMatchesUnstriped<PerGroupCodebook3<64>>(
            256, 512, 8, 20260902u, 70u * 1024, 64, 64 );
    }

    // 200 rows does not divide into whole strips: three strips of 80 leaves a trailing 40,
    // which is the case that needs its own cuBLASLt plan and a shorter final dequantize.
    TEST( CodebookLinearOpCuda, TwoBitStripedPrefillHandlesARaggedTrailingStrip )
    {
        runStripedPrefillMatchesUnstriped<PerGroupCodebook2<32>>(
            200, 512, 4, 20260903u, 70u * 1024, 80, 40 );
    }

    TEST( CodebookLinearOpCuda, ThreeBitStripedPrefillHandlesARaggedTrailingStrip )
    {
        runStripedPrefillMatchesUnstriped<PerGroupCodebook3<64>>(
            200, 512, 8, 20260904u, 70u * 1024, 80, 40 );
    }

    namespace
    {
        /**
         * @brief The same claim on the W4A8-FP8 prefill, which is a DIFFERENT striped path.
         *
         * FP4 does not reach runStagedPrefill: with kUseFp8ActivationPrefill on it takes the
         * W4A8 path, which expands to FP8 rather than BF16 and had its own unstriped
         * expansion of the whole weight matrix until 2026-08-26. The language-model head is
         * the shape that made that fatal -- 248320 x 5120 is 1212.5 MiB of staging asked for
         * whatever the row count -- and it went unseen because prefill evaluates the head at
         * exactly one position, which takes the decode matvec and never arrives here.
         *
         * Bitwise, not a tolerance, for the reason the codebook version gives: each output
         * channel is a dot product over exactly one strip and is never summed across strips,
         * so striping may move where the expanded weights live and must not touch the
         * arithmetic. A tolerance would also pass on a version that split the CONTRACTION,
         * which is the mistake this shape of striping exists to avoid.
         *
         * Weights are quantized by the op's own kernel rather than packed here: the nibble
         * order and scale convention are defined only by that kernel, and a second definition
         * in a test would be a second thing to keep right.
         */
        void runFp8StripedPrefillMatchesUnstriped(
            dim_t rows, dim_t columns, unsigned seed,
            std::size_t stagingCap, dim_t expectedStripRows, dim_t expectedTrailingRows )
        {
            using Fp4Policy = PerGroupFp4<128>;
            using Fp4Op = CudaLinearOp<TensorDataType::BF16, Fp4Policy>;

            constexpr dim_t kGroupSize = Fp4Policy::kQuantizationGroupSize;
            constexpr dim_t kBatchRows = 16;

            auto context = makeContextOrSkip();

            if ( context == nullptr )
                GTEST_SKIP() << "CUDA device not available";

            std::mt19937 generator( seed );
            std::normal_distribution<float> distribution( 0.0f, 1.0f );

            std::vector<std::uint16_t> weight_bits(
                static_cast<std::size_t>( rows ) * static_cast<std::size_t>( columns ) );

            for ( auto& bits : weight_bits )
            {
                bits = static_cast<std::uint16_t>(
                    std::bit_cast<std::uint32_t>( distribution( generator ) ) >> 16 );
            }

            const std::size_t weight_bytes = weight_bits.size() * sizeof( std::uint16_t );

            HostFp32 host_input( Device::Cpu(), shape_t{ kBatchRows, columns } );

            for ( dim_t m = 0; m < kBatchRows; ++m )
                for ( dim_t k = 0; k < columns; ++k )
                    host_input.data()[ m * columns + k ] = distribution( generator );

            DeviceBf16 batch_input( Device::Cuda( 0 ), shape_t{ kBatchRows, columns } );
            copy( host_input, batch_input, context.get() );
            context->synchronize();

            const auto runWithCap = [&]( std::size_t cap, dim_t* stripRows, dim_t* trailingRows )
            {
                LinearConfig config( columns, rows );
                config.withBias( false );

                Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource> packed(
                    Device::Cuda( 0 ), shape_t{ rows, columns / 2 } );
                Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> scales(
                    Device::Cuda( 0 ), shape_t{ rows, columns / kGroupSize } );

                Fp4Op op( context.get(), config, cap );

                op.setParameters( &packed, nullptr );
                op.setWeightScales( &scales );
                op.build( BuildContext( shape_t{ kBatchRows, columns }, RuntimeMode::Inference, false ) );

                *stripRows = op.getStripRows();
                *trailingRows = op.getTrailingStripRows();

                // After build, as Linear does it: quantize() writes the packed codes, the
                // group scales AND the static FP8 weight scale the plan's A_SCALE points at.
                Serialization::TensorMetadata meta{
                    TensorDataType::BF16, shape_t{ rows, columns }, weight_bytes };
                Serialization::TensorBlobView blob( meta, weight_bits.data(), weight_bytes );

                op.quantize( blob, packed, scales, shape_t{ rows, columns } );
                context->synchronize();

                DeviceBf16 batch_output( Device::Cuda( 0 ), shape_t{ kBatchRows, rows } );
                op.forward( batch_input, batch_output );
                context->synchronize();

                auto host = toHost<TensorDataType::FP32>( batch_output, context.get() );
                context->synchronize();

                return std::vector<float>( host.data(),
                    host.data() + static_cast<std::size_t>( kBatchRows * rows ) );
            };

            dim_t wholeStrip = 0, wholeTrailing = 0;
            const std::vector<float> unstriped =
                runWithCap( std::size_t( 1 ) << 40, &wholeStrip, &wholeTrailing );

            EXPECT_EQ( wholeStrip, rows ) << "a cap this large must not stripe at all";

            dim_t stripRows = 0, trailingRows = 0;
            const std::vector<float> striped = runWithCap( stagingCap, &stripRows, &trailingRows );

            EXPECT_EQ( stripRows, expectedStripRows );
            EXPECT_EQ( trailingRows, expectedTrailingRows );
            ASSERT_GT( rows, stripRows ) << "the cap must actually force more than one strip";

            ASSERT_EQ( unstriped.size(), striped.size() );

            for ( dim_t m = 0; m < kBatchRows; ++m )
                for ( dim_t row = 0; row < rows; ++row )
                {
                    const std::size_t index = static_cast<std::size_t>( m * rows + row );

                    EXPECT_EQ( std::bit_cast<std::uint32_t>( striped[ index ] ),
                        std::bit_cast<std::uint32_t>( unstriped[ index ] ) )
                        << "batch row " << m << ", output " << row
                        << " (strip " << ( row / stripRows ) << ")";
                }
        }
    }

    // 256 x 512 is 256 KiB staged whole in BF16 accounting; a 70 KiB cap asks for four
    // strips and divides evenly.
    TEST( CodebookLinearOpCuda, Fp4StripedFp8PrefillMatchesUnstriped )
    {
        runFp8StripedPrefillMatchesUnstriped( 256, 512, 20260905u, 70u * 1024, 64, 64 );
    }

    // 208 rows over strips of 80 leaves a trailing 48 -- the case needing its own plan and a
    // shorter final expansion, and the one where an output_row_stride mistake shows up as
    // channels written to the wrong columns rather than as a crash.
    TEST( CodebookLinearOpCuda, Fp4StripedFp8PrefillHandlesARaggedTrailingStrip )
    {
        runFp8StripedPrefillMatchesUnstriped( 208, 512, 20260906u, 70u * 1024, 80, 48 );
    }

    TEST( CodebookLinearOpCuda, TwoBitPrefillAndDecodeMatchCodec )
    {
        runPrefillAndDecodeAgainstCodec<PerGroupCodebook2<32>>( 256, 512, 4, 20260825u );
    }

    TEST( CodebookLinearOpCuda, ThreeBitPrefillAndDecodeMatchCodec )
    {
        runPrefillAndDecodeAgainstCodec<PerGroupCodebook3<64>>( 256, 3072, 8, 20260827u );
    }

    namespace
    {
        // Decode is bandwidth-bound, so the only sanity worth running is achieved
        // bandwidth: bytes the kernel must read, over the time it took. A codebook GEMV
        // that reads half the bytes of the qfp4 kernel and is not roughly twice as fast
        // has a problem the correctness tests cannot see.
        struct DecodeTiming
        {
            double milliseconds;
            double gigabytesPerSecond;
        };

        // The synchronize is a caller-supplied callable, not a context, because a
        // component constructed from a Device owns a stream of its own. Synchronizing the
        // test's context instead measured 4044 GB/s on a 504 GB/s card -- launch latency,
        // with every kernel still queued.
        template<typename TCallable, typename TSynchronize>
        DecodeTiming timeDecode( std::size_t weightBytes, TCallable&& launch,
            TSynchronize&& synchronize )
        {
            constexpr int kWarmup = 20;
            constexpr int kIterations = 200;

            for ( int i = 0; i < kWarmup; ++i )
                launch();

            synchronize();

            const auto start = std::chrono::steady_clock::now();

            for ( int i = 0; i < kIterations; ++i )
                launch();

            synchronize();

            const auto finish = std::chrono::steady_clock::now();
            const double seconds =
                std::chrono::duration<double>( finish - start ).count() / kIterations;

            return DecodeTiming{ seconds * 1000.0,
                static_cast<double>( weightBytes ) / seconds / 1e9 };
        }
    }

    // Not part of the correctness gate and not run in CI. Invoke with
    // --gtest_also_run_disabled_tests --gtest_filter=CodebookLinearOpCuda.DISABLED_DecodeBandwidth
    TEST( CodebookLinearOpCuda, DISABLED_DecodeBandwidth )
    {
        auto context = makeContextOrSkip();

        if ( context == nullptr )
            GTEST_SKIP() << "CUDA device not available";

        // A Llama 3.2 3B FFN projection -- the shape Phase 1 validates against.
        constexpr dim_t kColumns = 3072;
        constexpr dim_t kRows = 8192;

        DeviceBf16 input( Device::Cuda( 0 ), shape_t{ 1, kColumns } );
        DeviceBf16 output( Device::Cuda( 0 ), shape_t{ 1, kRows } );

        std::printf( "\n  %-22s %10s %10s %10s\n", "path", "bytes/MB", "ms", "GB/s" );

        const auto runCodebook = [&]<typename TPolicy>( const char* label, int entries )
        {
            constexpr dim_t kGroupSize = TPolicy::kQuantizationGroupSize;

            const PackedTensor packed =
                makePackedTensor( kRows, kColumns, kGroupSize, entries, 20260828u );

            LinearConfig config( kColumns, kRows );
            config.withBias( false );

            CodebookOpFor<TPolicy> op( context.get(), config );
            // Held for the duration of the timing loop: the operation borrows these.
            auto tensors = uploadWeights<TPolicy>( packed, kRows, kColumns );
            bindWeights<TPolicy>( op, tensors );
            op.build( BuildContext( shape_t{ 1, kColumns }, RuntimeMode::Inference, false ) );

            const std::size_t codeBytes = static_cast<std::size_t>( kRows )
                * static_cast<std::size_t>( kColumns ) * TPolicy::kCodeBits / 8;
            const std::size_t scaleBytes = static_cast<std::size_t>( kRows )
                * static_cast<std::size_t>( kColumns / kGroupSize ) * sizeof( std::uint16_t );
            const std::size_t weightBytes = codeBytes + scaleBytes;

            const DecodeTiming timing = timeDecode( weightBytes,
                [&] { op.forward( input, output ); },
                [&] { context->synchronize(); } );

            std::printf( "  %-22s %10.1f %10.4f %10.1f\n", label,
                static_cast<double>( weightBytes ) / 1e6,
                timing.milliseconds, timing.gigabytesPerSecond );
        };

        runCodebook.template operator()<PerGroupCodebook2<32>>( "codebook2 g32 (W2A16)", 4 );
        runCodebook.template operator()<PerGroupCodebook3<64>>( "codebook3 g64 (W3A16)", 8 );

        // The production FP4 decode matvec, reached through Linear's public surface rather
        // than its private kernel header. Component dispatch is included in the number;
        // at 12 MB of weight traffic per call that overhead is noise.
        {
            using QuantizedLinear = Mila::Dnn::Linear<DeviceType::Cuda, TensorDataType::BF16,
                Mila::Dnn::Quant::Weight::PerGroupFp4<128>>;

            std::vector<std::uint16_t> weight_bits(
                static_cast<std::size_t>( kRows * kColumns ), 0x3F00u );

            LinearConfig config( kColumns, kRows );
            config.withBias( false );

            QuantizedLinear linear( "bench_fp4", config, Device::Cuda( 0 ) );
            linear.build( BuildContext( shape_t{ 1, kColumns }, RuntimeMode::Inference, false ) );

            Serialization::TensorMetadata meta{ TensorDataType::BF16,
                shape_t{ kRows, kColumns }, weight_bits.size() * sizeof( std::uint16_t ) };
            Serialization::TensorBlobView blob( meta, weight_bits.data(),
                weight_bits.size() * sizeof( std::uint16_t ) );
            linear.loadParameter( "weight", blob );
            context->synchronize();

            const std::size_t weightBytes =
                static_cast<std::size_t>( kRows ) * static_cast<std::size_t>( kColumns ) / 2
                + static_cast<std::size_t>( kRows ) * ( kColumns / 128 ) * sizeof( float );

            const DecodeTiming timing = timeDecode( weightBytes,
                [&] { linear.forward( input ); },
                [&] { linear.synchronize(); } );

            std::printf( "  %-22s %10.1f %10.4f %10.1f\n", "fp4 g128 (W4A16)",
                static_cast<double>( weightBytes ) / 1e6,
                timing.milliseconds, timing.gigabytesPerSecond );
        }

        std::printf( "\n" );
    }

    namespace
    {
        // The Phase 1 exit gate: the packed tensors reach the operation through the
        // component's public load path rather than through setParameters(), which is the
        // only route a real artifact has. Everything the op tests drive by hand --
        // allocation, binding, the codebook and plane companions -- is Linear's job here.
        template<typename TPolicy>
        void runLoadParameterAndCompare(
            dim_t rows, dim_t columns, int entries, unsigned seed )
        {
            constexpr dim_t kGroupSize = TPolicy::kQuantizationGroupSize;

            auto context = makeContextOrSkip();

            if ( context == nullptr )
                GTEST_SKIP() << "CUDA device not available";

            const PackedTensor packed =
                makePackedTensor( rows, columns, kGroupSize, entries, seed );
            const std::vector<float> expected =
                referenceRowSums( packed, rows, columns, kGroupSize );

            LinearConfig config( columns, rows );
            config.withBias( false );

            Mila::Dnn::Linear<DeviceType::Cuda, TensorDataType::BF16, TPolicy> linear(
                "codebook_load", config, Device::Cuda( 0 ) );
            linear.build( BuildContext( shape_t{ 1, columns }, RuntimeMode::Inference, false ) );

            const auto load = [&]( const char* name, TensorDataType dtype,
                const shape_t& shape, const void* bytes, std::size_t nbytes )
            {
                Serialization::TensorMetadata meta{ dtype, shape, nbytes };
                Serialization::TensorBlobView blob( meta, bytes, nbytes );
                linear.loadParameter( name, blob );
            };

            // Names and dtypes exactly as quality_gate.py emits them.
            load( "weight", TensorDataType::UINT8, shape_t{ rows, columns / 4 },
                packed.planeTwoBits.data(), packed.planeTwoBits.size() );

            if constexpr ( TPolicy::kHasHighBitPlane )
            {
                load( "weight_high_plane", TensorDataType::UINT8, shape_t{ rows, columns / 8 },
                    packed.planeOneBit.data(), packed.planeOneBit.size() );
            }

            load( "weight_scale", TensorDataType::FP16,
                shape_t{ rows, groupsPerRow( columns, kGroupSize ) },
                packed.scaleBits.data(), packed.scaleBits.size() * sizeof( std::uint16_t ) );

            load( "weight_codebook", TensorDataType::FP32,
                shape_t{ static_cast<dim_t>( TPolicy::kCodebookEntries ) },
                packed.codebook.data(), packed.codebook.size() * sizeof( float ) );

            linear.synchronize();

            HostFp32 host_input( Device::Cpu(), shape_t{ 1, columns } );

            for ( dim_t column = 0; column < columns; ++column )
                host_input.data()[column] = packed.activations[static_cast<std::size_t>( column )];

            DeviceBf16 device_input( Device::Cuda( 0 ), shape_t{ 1, columns } );
            copy( host_input, device_input, context.get() );
            context->synchronize();

            const auto& device_output = linear.forward( device_input );
            linear.synchronize();

            auto host_output = toHost<TensorDataType::FP32>( device_output, context.get() );
            context->synchronize();

            for ( dim_t row = 0; row < rows; ++row ) {
                const float reference = expected[static_cast<std::size_t>( row )];
                const float tolerance = 0.01f + 0.005f * std::abs( reference );
                EXPECT_NEAR( host_output.data()[row], reference, tolerance ) << "row " << row;
            }
        }
    }

    TEST( CodebookLinearOpCuda, TwoBitLoadsThroughLoadParameter )
    {
        runLoadParameterAndCompare<PerGroupCodebook2<32>>( 64, 512, 4, 20260823u );
    }

    TEST( CodebookLinearOpCuda, ThreeBitLoadsThroughLoadParameter )
    {
        runLoadParameterAndCompare<PerGroupCodebook3<64>>( 48, 3072, 8, 20260824u );
    }

    TEST( CodebookLinearOpCuda, RejectsAWeightTensorAndAMismatchedPlane )
    {
        auto context = makeContextOrSkip();

        if ( context == nullptr )
            GTEST_SKIP() << "CUDA device not available";

        constexpr dim_t kRows = 16;
        constexpr dim_t kColumns = 128;

        const PackedTensor packed = makePackedTensor( kRows, kColumns, 32, 4, 20260826u );

        LinearConfig config( kColumns, kRows );
        config.withBias( false );

        CodebookOpFor<PerGroupCodebook2<32>> op( context.get(), config );
        auto tensors = uploadWeights<PerGroupCodebook2<32>>( packed, kRows, kColumns );

        EXPECT_THROW( op.setParameters( nullptr, nullptr ), std::invalid_argument );

        // A 2-bit policy handed a high-bit plane means the caller packed for the other
        // format; taking it would decode every code wrong and still return numbers.
        EXPECT_THROW( op.setCodebookTensors( &tensors.codebook, &tensors.highPlane ),
            std::invalid_argument );

        // Building before the buffers are bound cannot produce a diagnosable failure later.
        EXPECT_THROW(
            op.build( BuildContext( shape_t{ 1, kColumns }, RuntimeMode::Inference, false ) ),
            std::runtime_error );
    }
}
