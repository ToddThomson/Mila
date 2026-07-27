/**
 * @file Residual.Cuda.cpp
 * @brief Concrete-component tests for Residual<DeviceType::Cuda, {FP32, BF16}>.
 *
 * CUDA companion to Residual.Cpu.cpp: a TYPED_TEST sweep over FP32 and BF16
 * (see Linear.Cuda.cpp for the precision-sweep reference). Residual computes
 * out = a + b; backward routes the upstream gradient to both inputs.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

import Mila;
import Compute.ExecutionContext;

namespace Mila::Tests::Dnn::Components::Connections
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float atol = 1e-3f;
            static constexpr float rtol = 1e-4f;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr float atol = 5e-2f;
            static constexpr float rtol = 5e-2f;
            static constexpr const char* name = "Bf16";
        };

        using ResidualPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

        class PrecisionNames
        {
        public:
            template<typename TPrecisionTag>
            static std::string GetName( int )
            {
                return TPrecisionTag::name;
            }
        };
    }

    template<typename TPrecisionTag>
    class ResidualCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using ResidualType = Mila::Dnn::Residual<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( ResidualType::getDeviceType() == DeviceType::Cuda );
        static_assert( ResidualType::getPrecision() == P );

        void SetUp() override
        {
            try
            {
                cuda_context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                cuda_context_ = nullptr;
            }

            if ( !cuda_context_ )
            {
                GTEST_SKIP() << "CUDA device not available";
            }
        }

        std::unique_ptr<ResidualType> builtResidual( const shape_t& shape, RuntimeMode mode )
        {
            auto residual = std::make_unique<ResidualType>( "residual", ResidualConfig(), Device::Cuda( 0 ) );
            residual->build( BuildContext( shape, mode, false ) );

            return residual;
        }

        HostFp32 rampHost( const shape_t& shape, float start, float step )
        {
            HostFp32 host( Device::Cpu(), shape );
            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = start + step * static_cast<float>( i );
            }

            return host;
        }

        DeviceTensor toDevice( const HostFp32& host )
        {
            DeviceTensor device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        HostFp32 toFloat( const DeviceTensor& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, cuda_context_.get() );
            cuda_context_->synchronize();

            return host;
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TYPED_TEST_SUITE( ResidualCudaTests, ResidualPrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction
    // ====================================================================

    TYPED_TEST( ResidualCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::ResidualType residual( "residual", ResidualConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( residual.getDeviceId().type, DeviceType::Cuda );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TYPED_TEST( ResidualCudaTests, Forward_MatchesSum )
    {
        const shape_t shape{ 2, 3, 4 };
        auto residual = this->builtResidual( shape, RuntimeMode::Inference );

        auto host_a = this->rampHost( shape, -1.0f, 0.1f );
        auto host_b = this->rampHost( shape, 0.5f, -0.05f );
        auto device_a = this->toDevice( host_a );
        auto device_b = this->toDevice( host_b );

        auto& device_out = residual->forward( device_a, device_b );
        residual->synchronize();

        auto out = this->toFloat( device_out );
        auto a = this->toFloat( device_a );
        auto b = this->toFloat( device_b );

        ASSERT_EQ( out.size(), a.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const float expected = a.data()[ i ] + b.data()[ i ];
            const float tolerance = TypeParam::atol + TypeParam::rtol * std::fabs( expected );

            EXPECT_NEAR( out.data()[ i ], expected, tolerance ) << "sum mismatch at " << i;
        }
    }

    // ====================================================================
    // F. Backward (gradient routing)
    // ====================================================================

    TYPED_TEST( ResidualCudaTests, Backward_RoutesGradientToBothInputs )
    {
        const shape_t shape{ 2, 3, 4 };
        auto residual = this->builtResidual( shape, RuntimeMode::Training );

        auto device_a = this->toDevice( this->rampHost( shape, -1.0f, 0.1f ) );
        auto device_b = this->toDevice( this->rampHost( shape, 0.5f, -0.05f ) );
        auto host_grad = this->rampHost( shape, 0.2f, 0.03f );
        auto device_grad = this->toDevice( host_grad );

        residual->forward( device_a, device_b );
        auto [da_dev, db_dev] = residual->backward( device_a, device_b, device_grad );
        residual->synchronize();

        auto da = this->toFloat( da_dev );
        auto db = this->toFloat( db_dev );
        auto grad = this->toFloat( device_grad );

        ASSERT_EQ( da.size(), grad.size() );

        for ( dim_t i = 0; i < grad.size(); ++i )
        {
            const float tolerance = TypeParam::atol + TypeParam::rtol * std::fabs( grad.data()[ i ] );
            EXPECT_NEAR( da.data()[ i ], grad.data()[ i ], tolerance ) << "da mismatch at " << i;
            EXPECT_NEAR( db.data()[ i ], grad.data()[ i ], tolerance ) << "db mismatch at " << i;
        }
    }

    // ====================================================================
    // G. Parameters (stateless) & J. Type
    // ====================================================================

    TYPED_TEST( ResidualCudaTests, Parameters_AreEmpty )
    {
        auto residual = this->builtResidual( shape_t{ 2, 4 }, RuntimeMode::Inference );

        EXPECT_EQ( residual->parameterCount(), 0 );
        EXPECT_TRUE( residual->getParameters().empty() );
    }

    TYPED_TEST( ResidualCudaTests, GetType_IsResidual )
    {
        typename TestFixture::ResidualType residual( "residual", ResidualConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( residual.getType(), ComponentType::Residual );
    }
}
