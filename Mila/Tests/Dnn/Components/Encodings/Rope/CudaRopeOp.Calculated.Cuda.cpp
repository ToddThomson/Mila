/**
 * @file CudaRopeOp.Calculated.Cuda.cpp
 * @brief RETIRED 2026-10-01, out of the build, kept for reference: the gate that retired the RoPE table.
 *
 * It compared CudaRopeOp<P, true> (calculated) against CudaRopeOp<P, false> (table); the table and the template
 * axis are gone, so it no longer compiles. Its last run, on the final kernel, passed bit for bit; the result and
 * the cost measurement are recorded in MemoryFootprint.md.
 *
 * The calculated rotation (CudaRopeOp<P, true>) against the table it replaces.
 *
 * Both compute every angle through one device function, so they agree bit for bit at every position a model reaches,
 * at Llama's, Gemma's (local and global) and Qwen's geometries, through prefill and decode. The calculated op holds no
 * state. A disabled test times both, the measurement the table's retirement is decided on.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

import Mila;
import Compute.CudaRopeOp;

namespace Mila::Tests::Dnn::Components::Encodings::Rope::Calculated
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        struct Geometry
        {
            const char* name;
            dim_t heads;
            dim_t kv_heads;
            dim_t head_dim;
            dim_t rotary_dim;
            RotaryLayout layout;
            float base;
            RopeFrequencyScaling scaling;
            dim_t trained_maximum;
        };

        // The geometries the families run, at each model's trained maximum context.
        const std::vector<Geometry>& geometries()
        {
            static const std::vector<Geometry> all = {
                { "llama_3_1_8b", 32, 8, 128, 0, RotaryLayout::WholeHead, 500000.0f, RopeFrequencyScaling{ 8.0f, 1.0f, 4.0f, 8192 }, 131072 },
                { "gemma_4_local", 16, 8, 256, 0, RotaryLayout::WholeHead, 10000.0f, RopeFrequencyScaling{}, 262144 },
                { "gemma_4_global", 16, 1, 512, 128, RotaryLayout::WholeHead, 1000000.0f, RopeFrequencyScaling{}, 262144 },
                { "qwen_3_8", 24, 4, 256, 64, RotaryLayout::RotaryPrefix, 10000000.0f, RopeFrequencyScaling{}, 262144 },
            };

            return all;
        }

        RopeConfig configFor( const Geometry& geometry )
        {
            RopeConfig config( geometry.heads * geometry.head_dim, geometry.heads, geometry.kv_heads, geometry.trained_maximum );
            config.withBase( geometry.base )
                .withRotaryDim( geometry.rotary_dim )
                .withRotaryLayout( geometry.layout )
                .withFrequencyScaling( geometry.scaling );

            return config;
        }

        HostFp32 normalHost( const shape_t& shape, std::uint32_t seed )
        {
            HostFp32 host( Device::Cpu(), shape );
            std::mt19937 generator( seed );
            std::normal_distribution<float> normal( 0.0f, 1.0f );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = normal( generator );
            }

            return host;
        }
    }

    template<TensorDataType TPrecision>
    class CalculatedRopeHarness
    {
    public:
        using TableOp = Compute::Cuda::Rope::CudaRopeOp<TPrecision, false>;
        using CalculatedOp = Compute::Cuda::Rope::CudaRopeOp<TPrecision, true>;
        using DeviceTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

        explicit CalculatedRopeHarness( IExecutionContext* context ) : context_( context )
        {
        }

        /// Rotate one chunk (or one decode token) through an op in place, and return the output bytes of Q then K.
        template<typename TOp>
        std::vector<std::uint8_t> rotate( TOp& op, const Geometry& geometry, dim_t rows, dim_t position, bool decode )
        {
            DeviceTensor q( Device::Cuda( 0 ), shape_t{ 1, rows, geometry.heads * geometry.head_dim } );
            DeviceTensor k( Device::Cuda( 0 ), shape_t{ 1, rows, geometry.kv_heads * geometry.head_dim } );

            copy( normalHost( q.shape(), 11 ), q, context_ );
            copy( normalHost( k.shape(), 29 ), k, context_ );

            if ( decode )
            {
                context_->setDecodePosition( position );
                op.decode( q, k, q, k, position );
            }
            else
            {
                op.prefill( q, k, q, k, position );
            }

            context_->synchronize();

            return bytesOf( q, k );
        }

    private:

        // Widened to FP32 on the way back, which is exact for BF16, so equal bytes here are equal bits on the device.
        std::vector<std::uint8_t> bytesOf( const DeviceTensor& q, const DeviceTensor& k )
        {
            auto host_q = toHost<TensorDataType::FP32>( q, context_ );
            auto host_k = toHost<TensorDataType::FP32>( k, context_ );
            context_->synchronize();

            const auto* q_bytes = reinterpret_cast<const std::uint8_t*>( host_q.data() );
            const auto* k_bytes = reinterpret_cast<const std::uint8_t*>( host_k.data() );

            std::vector<std::uint8_t> bytes( q_bytes, q_bytes + static_cast<std::size_t>( host_q.size() ) * sizeof( float ) );
            bytes.insert( bytes.end(), k_bytes, k_bytes + static_cast<std::size_t>( host_k.size() ) * sizeof( float ) );

            return bytes;
        }

        IExecutionContext* context_;
    };

    class CudaRopeCalculatedTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            context_ = createExecutionContext( Device::Cuda( 0 ) );
        }

        template<TensorDataType TPrecision>
        void expectBitIdenticalAtEveryReach()
        {
            using Harness = CalculatedRopeHarness<TPrecision>;

            Harness harness( context_.get() );
            constexpr dim_t kChunk = 64;

            for ( const Geometry& geometry : geometries() )
            {
                const RopeConfig config = configFor( geometry );
                const BuildContext build( shape_t{ 1, geometry.trained_maximum }, RuntimeMode::Inference, false );

                typename Harness::TableOp table( context_.get(), config );
                typename Harness::CalculatedOp calculated( context_.get(), config );

                table.build( build );
                calculated.build( build );

                const dim_t last = geometry.trained_maximum - 1;
                const std::vector<dim_t> prefill_offsets = { 0, 1000, 65536 - kChunk, geometry.trained_maximum - kChunk };
                const std::vector<dim_t> decode_positions = { 0, 1, 4097, 105615, 105616, last };

                for ( const dim_t offset : prefill_offsets )
                {
                    EXPECT_EQ( harness.rotate( table, geometry, kChunk, offset, false ),
                        harness.rotate( calculated, geometry, kChunk, offset, false ) )
                        << geometry.name << " prefill of " << kChunk << " rows at offset " << offset;
                }

                for ( const dim_t position : decode_positions )
                {
                    EXPECT_EQ( harness.rotate( table, geometry, 1, position, true ),
                        harness.rotate( calculated, geometry, 1, position, true ) )
                        << geometry.name << " decode at " << position;
                }
            }
        }

        std::unique_ptr<IExecutionContext> context_;
    };

    TEST_F( CudaRopeCalculatedTests, Bf16_IsBitIdenticalToTheTable )
    {
        expectBitIdenticalAtEveryReach<TensorDataType::BF16>();
    }

    TEST_F( CudaRopeCalculatedTests, Fp32_IsBitIdenticalToTheTable )
    {
        expectBitIdenticalAtEveryReach<TensorDataType::FP32>();
    }

    TEST_F( CudaRopeCalculatedTests, HoldsNoState )
    {
        const Geometry& geometry = geometries()[ 2 ];
        const BuildContext build = BuildContext( shape_t{ 1, geometry.trained_maximum }, RuntimeMode::Inference, false )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );

        Compute::Cuda::Rope::CudaRopeOp<TensorDataType::BF16, true> calculated( context_.get(), configFor( geometry ) );

        EXPECT_EQ( calculated.getRequiredStateMemorySize( build ), 0u );

        calculated.build( build );

        EXPECT_EQ( calculated.getStateMemorySize(), 0u );
    }

    TEST_F( CudaRopeCalculatedTests, RefusesPositionsPastTheBuiltLength )
    {
        const Geometry& geometry = geometries()[ 0 ];
        constexpr dim_t kBuilt = 256;

        Compute::Cuda::Rope::CudaRopeOp<TensorDataType::BF16, true> calculated( context_.get(), configFor( geometry ) );
        calculated.build( BuildContext( shape_t{ 1, kBuilt }, RuntimeMode::Inference, false ) );

        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> q( Device::Cuda( 0 ), shape_t{ 1, 1, geometry.heads * geometry.head_dim } );
        Tensor<TensorDataType::BF16, CudaDeviceMemoryResource> k( Device::Cuda( 0 ), shape_t{ 1, 1, geometry.kv_heads * geometry.head_dim } );

        EXPECT_THROW( calculated.decode( q, k, q, k, kBuilt ), std::invalid_argument );
        EXPECT_THROW( calculated.prefill( q, k, q, k, kBuilt ), std::invalid_argument );
    }

    // Measurement, not a gate: what calculating costs against reading the table, per op call, at the chunk a 16 GB
    // card prefills with and at decode, near the start of the context and at its end, where cosf takes its slow
    // argument reduction (|angle| past 105615 radians).
    TEST_F( CudaRopeCalculatedTests, DISABLED_Cost_CalculatedAgainstTable )
    {
        constexpr int kWarmup = 20;
        constexpr int kIterations = 200;

        auto time = [&]( auto& op, const Geometry& geometry, dim_t rows, dim_t position, bool decode )
        {
            using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;

            DeviceBf16 q( Device::Cuda( 0 ), shape_t{ 1, rows, geometry.heads * geometry.head_dim } );
            DeviceBf16 k( Device::Cuda( 0 ), shape_t{ 1, rows, geometry.kv_heads * geometry.head_dim } );

            copy( normalHost( q.shape(), 3 ), q, context_.get() );
            copy( normalHost( k.shape(), 5 ), k, context_.get() );
            context_->setDecodePosition( position );

            auto run = [&]
            {
                if ( decode )
                    op.decode( q, k, q, k, position );
                else
                    op.prefill( q, k, q, k, position );
            };

            for ( int i = 0; i < kWarmup; ++i )
                run();

            context_->synchronize();

            const auto start = std::chrono::steady_clock::now();

            for ( int i = 0; i < kIterations; ++i )
                run();

            context_->synchronize();

            const auto elapsed = std::chrono::steady_clock::now() - start;

            return std::chrono::duration<double, std::micro>( elapsed ).count() / kIterations;
        };

        for ( const Geometry& geometry : geometries() )
        {
            const RopeConfig config = configFor( geometry );
            const BuildContext build( shape_t{ 1, geometry.trained_maximum }, RuntimeMode::Inference, false );

            Compute::Cuda::Rope::CudaRopeOp<TensorDataType::BF16, false> table( context_.get(), config );
            Compute::Cuda::Rope::CudaRopeOp<TensorDataType::BF16, true> calculated( context_.get(), config );

            table.build( build );
            calculated.build( build );

            for ( const dim_t rows : { dim_t{ 1024 }, dim_t{ 128 } } )
            {
                for ( const dim_t offset : { dim_t{ 0 }, geometry.trained_maximum - rows } )
                {
                    const double table_us = time( table, geometry, rows, offset, false );
                    const double calculated_us = time( calculated, geometry, rows, offset, false );

                    std::cout << std::format( "[rope cost] {:<15} prefill {:>4} rows at {:>6}: table {:7.2f} us, calculated {:7.2f} us ({:+.2f})\n",
                        geometry.name, rows, offset, table_us, calculated_us, calculated_us - table_us );
                }
            }

            for ( const dim_t position : { dim_t{ 100 }, geometry.trained_maximum - 1 } )
            {
                const double table_us = time( table, geometry, 1, position, true );
                const double calculated_us = time( calculated, geometry, 1, position, true );

                std::cout << std::format( "[rope cost] {:<15} decode at {:>6}: table {:7.2f} us, calculated {:7.2f} us ({:+.2f})\n",
                    geometry.name, position, table_us, calculated_us, calculated_us - table_us );
            }
        }
    }
}
