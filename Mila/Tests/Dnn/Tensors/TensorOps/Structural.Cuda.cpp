/**
 * @file Structural.Cuda.cpp
 * @brief CUDA tests for TensorOps split() (Dnn.TensorOps:Structural).
 *
 * split() is CUDA-only: there is no CpuTensorOps::split, so unlike every other
 * TensorOps area this has no .Cpu.cpp companion -- a CPU instantiation would not
 * compile at all. Both overloads (2-way and 3-way last-dimension partition) are
 * covered here together with every documented precondition throw.
 *
 * Values move through copy()/toHost() rather than data(): a device tensor's
 * host_type is float for every float dtype, so indexing data() on a BF16 tensor
 * would write 4-byte floats over 2-byte storage. The transfer ops convert.
 *
 * Test values are small exact integers -- the largest is 192, inside BF16's
 * 8-bit mantissa -- so the same expectations hold at both precisions and a
 * mismatch means misrouted data, never rounding.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>   // cudaDeviceSynchronize -- the default-stream case below
#include <memory>
#include <vector>

import Mila;
import Compute.ExecutionContext;

namespace Mila::Tests::Dnn::Tensors::TensorOps
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class TensorOpsStructuralCudaTests : public testing::Test
    {
    protected:
        void SetUp() override
        {
            try
            {
                context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                context_ = nullptr;
            }
        }

        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // value(b, t, d) = ((b * T + t) * D + d) -- distinct per element, so a
        // misrouted slice is identifiable rather than merely unequal.
        static HostFp32 ramp( dim_t B, dim_t T, dim_t D )
        {
            HostFp32 host( Device::Cpu(), shape_t{ B, T, D } );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = static_cast<float>( i );
            }

            return host;
        }

        template<TensorDataType TDataType>
        Tensor<TDataType, CudaDeviceMemoryResource> toDevice( const HostFp32& host )
        {
            Tensor<TDataType, CudaDeviceMemoryResource> device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, context_.get() );
            context_->synchronize();

            return device;
        }

        template<TensorDataType TDataType>
        std::vector<float> toHostVector( const Tensor<TDataType, CudaDeviceMemoryResource>& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, context_.get() );
            context_->synchronize();

            return std::vector<float>( host.data(), host.data() + host.size() );
        }

        // The expected contents of the slice covering [offset, offset + width) of
        // each row of a [B, T, D] ramp.
        static std::vector<float> expectedSlice(
            dim_t B, dim_t T, dim_t D, dim_t offset, dim_t width )
        {
            std::vector<float> expected;
            expected.reserve( static_cast<size_t>( B * T * width ) );

            for ( dim_t row = 0; row < B * T; ++row )
            {
                for ( dim_t d = 0; d < width; ++d )
                {
                    expected.push_back( static_cast<float>( row * D + offset + d ) );
                }
            }

            return expected;
        }

        std::unique_ptr<IExecutionContext> context_;
    };

#define SKIP_WITHOUT_CUDA()                                            \
    if ( context_ == nullptr )                                         \
    {                                                                  \
        GTEST_SKIP() << "CUDA device not available.";                  \
    }

    // ====================================================================
    // A. Partitioning -- the happy path
    // ====================================================================

    TEST_F( TensorOpsStructuralCudaTests, Split2_PartitionsLastDimension )
    {
        SKIP_WITHOUT_CUDA();

        constexpr dim_t B = 2, T = 3, D0 = 8, D1 = 16, D = D0 + D1;

        auto input = toDevice<TensorDataType::FP32>( ramp( B, T, D ) );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ B, T, D0 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ B, T, D1 } );

        split( input, out0, out1, context_.get() );
        context_->synchronize();

        // out1 matching exactly is also the guard on the 2-way overload's aliasing
        // trick: it passes out1 as the kernel's third output with a zero-width
        // slice, so any write through that aliased pointer would corrupt out1 here.
        EXPECT_EQ( toHostVector( out0 ), expectedSlice( B, T, D, 0, D0 ) );
        EXPECT_EQ( toHostVector( out1 ), expectedSlice( B, T, D, D0, D1 ) );
    }

    TEST_F( TensorOpsStructuralCudaTests, Split3_PartitionsLastDimension )
    {
        SKIP_WITHOUT_CUDA();

        // The fused-QKV shape this op exists for: three unequal slices.
        constexpr dim_t B = 2, T = 3, D0 = 4, D1 = 8, D2 = 12, D = D0 + D1 + D2;

        auto input = toDevice<TensorDataType::FP32>( ramp( B, T, D ) );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ B, T, D0 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ B, T, D1 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out2( Device::Cuda( 0 ), shape_t{ B, T, D2 } );

        split( input, out0, out1, out2, context_.get() );
        context_->synchronize();

        EXPECT_EQ( toHostVector( out0 ), expectedSlice( B, T, D, 0, D0 ) );
        EXPECT_EQ( toHostVector( out1 ), expectedSlice( B, T, D, D0, D1 ) );
        EXPECT_EQ( toHostVector( out2 ), expectedSlice( B, T, D, D0 + D1, D2 ) );
    }

    // BF16 is a separate kernel moving 8 elements per 16-byte vector where FP32
    // moves 4, so it is a distinct code path, not a dtype sweep of the same one.
    TEST_F( TensorOpsStructuralCudaTests, Split3_Bf16_PartitionsLastDimension )
    {
        SKIP_WITHOUT_CUDA();

        constexpr dim_t B = 2, T = 3, D0 = 8, D1 = 8, D2 = 16, D = D0 + D1 + D2;

        auto input = toDevice<TensorDataType::BF16>( ramp( B, T, D ) );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ B, T, D0 } );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ B, T, D1 } );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> out2( Device::Cuda( 0 ), shape_t{ B, T, D2 } );

        split( input, out0, out1, out2, context_.get() );
        context_->synchronize();

        EXPECT_EQ( toHostVector( out0 ), expectedSlice( B, T, D, 0, D0 ) );
        EXPECT_EQ( toHostVector( out1 ), expectedSlice( B, T, D, D0, D1 ) );
        EXPECT_EQ( toHostVector( out2 ), expectedSlice( B, T, D, D0 + D1, D2 ) );
    }

    TEST_F( TensorOpsStructuralCudaTests, Split_NullExecutionContext_UsesDefaultStream )
    {
        SKIP_WITHOUT_CUDA();

        constexpr dim_t B = 1, T = 2, D0 = 4, D1 = 4, D = D0 + D1;

        auto input = toDevice<TensorDataType::FP32>( ramp( B, T, D ) );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ B, T, D0 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ B, T, D1 } );

        // exec_context defaults to nullptr, documented as running on the default stream.
        split( input, out0, out1 );
        cudaDeviceSynchronize();

        EXPECT_EQ( toHostVector( out0 ), expectedSlice( B, T, D, 0, D0 ) );
        EXPECT_EQ( toHostVector( out1 ), expectedSlice( B, T, D, D0, D1 ) );
    }

    // ====================================================================
    // B. Preconditions -- one test per documented throw
    // ====================================================================

    TEST_F( TensorOpsStructuralCudaTests, Split2_RankNotThree_Throws )
    {
        SKIP_WITHOUT_CUDA();

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input( Device::Cuda( 0 ), shape_t{ 6, 8 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ 6, 4 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ 6, 4 } );

        EXPECT_THROW( split( input, out0, out1, context_.get() ), std::invalid_argument );
    }

    TEST_F( TensorOpsStructuralCudaTests, Split2_OutputDimsDoNotSumToInput_Throws )
    {
        SKIP_WITHOUT_CUDA();

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input( Device::Cuda( 0 ), shape_t{ 2, 3, 16 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );

        EXPECT_THROW( split( input, out0, out1, context_.get() ), std::invalid_argument );
    }

    TEST_F( TensorOpsStructuralCudaTests, Split2_BatchOrTokenMismatch_Throws )
    {
        SKIP_WITHOUT_CUDA();

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input( Device::Cuda( 0 ), shape_t{ 2, 3, 8 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ 2, 5, 4 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );

        EXPECT_THROW( split( input, out0, out1, context_.get() ), std::invalid_argument );
    }

    TEST_F( TensorOpsStructuralCudaTests, Split3_RankNotThree_Throws )
    {
        SKIP_WITHOUT_CUDA();

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input( Device::Cuda( 0 ), shape_t{ 6, 12 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ 6, 4 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ 6, 4 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out2( Device::Cuda( 0 ), shape_t{ 6, 4 } );

        EXPECT_THROW( split( input, out0, out1, out2, context_.get() ), std::invalid_argument );
    }

    TEST_F( TensorOpsStructuralCudaTests, Split3_OutputDimsDoNotSumToInput_Throws )
    {
        SKIP_WITHOUT_CUDA();

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input( Device::Cuda( 0 ), shape_t{ 2, 3, 24 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out2( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );

        EXPECT_THROW( split( input, out0, out1, out2, context_.get() ), std::invalid_argument );
    }

    TEST_F( TensorOpsStructuralCudaTests, Split3_BatchOrTokenMismatch_Throws )
    {
        SKIP_WITHOUT_CUDA();

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input( Device::Cuda( 0 ), shape_t{ 2, 3, 12 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ 3, 3, 4 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out2( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );

        EXPECT_THROW( split( input, out0, out1, out2, context_.get() ), std::invalid_argument );
    }

    // ====================================================================
    // C. Vectorization alignment -- the precondition is dtype-dependent
    // ====================================================================

    // Each thread moves one 16-byte vector, so the required slice alignment is
    // 16 / sizeof(element): 4 elements for FP32.
    TEST_F( TensorOpsStructuralCudaTests, Split3_Fp32_SliceNotMultipleOfFour_Throws )
    {
        SKIP_WITHOUT_CUDA();

        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> input( Device::Cuda( 0 ), shape_t{ 2, 3, 12 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ 2, 3, 2 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ 2, 3, 6 } );
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> out2( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );

        EXPECT_THROW( split( input, out0, out1, out2, context_.get() ), std::invalid_argument );
    }

    // ...and 8 elements for BF16, whose kernel packs eight bf16 into the same
    // 16-byte uint4. A multiple-of-4-but-not-8 slice used to pass validation and
    // then run the kernel's D0/8 index arithmetic on a truncated quotient, storing
    // eight elements into a four-element output row. Regression guard for that.
    TEST_F( TensorOpsStructuralCudaTests, Split3_Bf16_SliceNotMultipleOfEight_Throws )
    {
        SKIP_WITHOUT_CUDA();

        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> input( Device::Cuda( 0 ), shape_t{ 2, 3, 16 } );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> out2( Device::Cuda( 0 ), shape_t{ 2, 3, 8 } );

        EXPECT_THROW( split( input, out0, out1, out2, context_.get() ), std::invalid_argument );
    }

    TEST_F( TensorOpsStructuralCudaTests, Split2_Bf16_SliceNotMultipleOfEight_Throws )
    {
        SKIP_WITHOUT_CUDA();

        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> input( Device::Cuda( 0 ), shape_t{ 2, 3, 8 } );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> out0( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> out1( Device::Cuda( 0 ), shape_t{ 2, 3, 4 } );

        EXPECT_THROW( split( input, out0, out1, context_.get() ), std::invalid_argument );
    }

#undef SKIP_WITHOUT_CUDA
}
