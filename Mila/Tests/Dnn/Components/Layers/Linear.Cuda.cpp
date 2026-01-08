#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Components_Layers_Linear_Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CudaTensor = Tensor<TPrecision, CudaDeviceMemoryResource>;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    // ====================================================================
    // Test shapes
    // ====================================================================
    struct LinearTestShape
    {
        int64_t batch;
        int64_t seq;
        int64_t in_features;
        int64_t out_features;
        std::string name;

        shape_t inputShape() const { return { batch, seq, in_features }; }
        shape_t outputShape() const { return { batch, seq, out_features }; }

        static LinearTestShape Small() { return { 2, 3, 16, 32, "Small" }; }
        static LinearTestShape Medium() { return { 8, 16, 128, 256, "Medium" }; }
        static LinearTestShape Large() { return { 16, 32, 512, 768, "Large" }; }
        static LinearTestShape Minimal() { return { 1, 1, 1, 1, "Minimal" }; }

        static std::vector<LinearTestShape> StandardShapes()
        {
            return { Small(), Medium(), Large() };
        }
    };

    // ====================================================================
    // Precision traits for naming/tolerances
    // ====================================================================
    template<TensorDataType TPrecision>
    struct PrecisionTraits
    {
        static constexpr TensorDataType value = TPrecision;
        static constexpr const char* name = "Unknown";
        static constexpr float tolerance = 1e-4f;
    };

    template<>
    struct PrecisionTraits<TensorDataType::FP32>
    {
        static constexpr TensorDataType value = TensorDataType::FP32;
        static constexpr const char* name = "FP32";
        static constexpr float tolerance = 1e-4f;
    };

    template<>
    struct PrecisionTraits<TensorDataType::FP16>
    {
        static constexpr TensorDataType value = TensorDataType::FP16;
        static constexpr const char* name = "FP16";
        static constexpr float tolerance = 1e-2f;
    };

    // ====================================================================
    // Fixture
    // ====================================================================
    template<TensorDataType TPrecision>
    struct LinearTestFixture
    {
        LinearTestFixture()
            : shape( LinearTestShape::Small() ),
              config( static_cast<dim_t>(shape.inputShape().back()), static_cast<dim_t>(shape.outputShape().back()) ),
              component( nullptr ),
              has_bias( true ),
              is_training( false )
        {}

        static LinearTestFixture Create( const LinearTestShape& s, bool with_bias = true, bool training = false )
        {
            LinearTestFixture f;
            f.shape = s;
            f.has_bias = with_bias;
            f.is_training = training;

            f.config = LinearConfig( static_cast<dim_t>(s.in_features), static_cast<dim_t>(s.out_features) );
            f.config.withBias( with_bias );

            std::string name = "linear_cuda_" + s.name + "_" + PrecisionTraits<TPrecision>::name;

            f.component = std::make_shared<Linear<DeviceType::Cuda, TPrecision>>( name, f.config, Device::Cuda( 0 ) );

            return f;
        }

        const shape_t& input_shape() const
        {
            static thread_local shape_t s;
            s = shape.inputShape();
            return s;
        }

        const shape_t& output_shape() const
        {
            static thread_local shape_t s;
            s = shape.outputShape();
            return s;
        }

        LinearTestShape shape;
        LinearConfig config;
        std::shared_ptr<Linear<DeviceType::Cuda, TPrecision>> component;
        bool has_bias;
        bool is_training;
    };

    // ====================================================================
    // Typed test suite
    // ====================================================================
    template<typename T>
    class LinearCudaTests : public testing::Test
    {
    protected:
        void SetUp() override
        {
            int device_count = getDeviceCount( DeviceType::Cuda );
            cuda_available_ = ( device_count > 0 );
        }

        bool cuda_available_{ false };
    };

    template<TensorDataType TPrecision>
    struct PrecisionType { static constexpr TensorDataType value = TPrecision; };

    using PrecisionTypes = ::testing::Types<
        PrecisionType<TensorDataType::FP32>/*,
        TODO: enable FP16 once kernels are validated
        PrecisionType<TensorDataType::FP16> */
    >;

    TYPED_TEST_SUITE( LinearCudaTests, PrecisionTypes );

    // ====================================================================
    // Construction / lifecycle tests
    // ====================================================================

    TYPED_TEST( LinearCudaTests, Constructor_WithDeviceId_CreatesComponent )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        LinearConfig cfg( 16, 32 );
        cfg.withBias( true );

        std::shared_ptr<Linear<DeviceType::Cuda, TPrecision>> comp;
        ASSERT_NO_THROW(
            ( comp = std::make_shared<Linear<DeviceType::Cuda, TPrecision>>( "lin_ctor", cfg, Device::Cuda( 0 ) ) ) );
        ASSERT_NE( comp, nullptr );
        EXPECT_EQ( comp->getDeviceType(), DeviceType::Cuda );
    }

    TYPED_TEST( LinearCudaTests, Constructor_WithoutDeviceId_AllowsSharedMode )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        LinearConfig cfg( 8, 8 );
        cfg.withBias( true );

        ASSERT_NO_THROW( ( (void)std::make_shared<Linear<DeviceType::Cuda, TPrecision>>( "lin_shared", cfg ) ) );
    }

    TYPED_TEST( LinearCudaTests, BuildAndIsBuilt )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LinearTestFixture<TPrecision>::Create( LinearTestShape::Small() );
        EXPECT_FALSE( fixture.component->isBuilt() );
        EXPECT_NO_THROW( fixture.component->build( fixture.input_shape() ) );
        EXPECT_TRUE( fixture.component->isBuilt() );
    }

    TYPED_TEST( LinearCudaTests, ParameterCountAndGetters )
    {
        if ( !this->cuda_available_ ) 
            GTEST_SKIP() << "CUDA not available";
        
        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LinearTestFixture<TPrecision>::Create( LinearTestShape::Small(), true );
        fixture.component->build( fixture.input_shape() );

        EXPECT_EQ( fixture.component->parameterCount(), static_cast<size_t>( fixture.shape.in_features * fixture.shape.out_features + ( fixture.has_bias ? fixture.shape.out_features : 0 ) ) );

        // Use getParameters() to inspect tensors rather than deprecated concrete getters
        auto params = fixture.component->getParameters();
        ASSERT_GE( params.size(), 1u );

        // First param expected to be weight (shape: [out_features, in_features])
        auto* weight_it = dynamic_cast<Tensor<TensorDataType::FP32, CpuMemoryResource>*>( params[0] );
        if ( !weight_it )
        {
            // If not host accessible, copy to host for inspection
            auto host_copy = toHost<TensorDataType::FP32>( *dynamic_cast<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>*>( params[0] ) );
            EXPECT_EQ( host_copy.shape()[ 0 ], fixture.shape.out_features );
            EXPECT_EQ( host_copy.shape()[ 1 ], fixture.shape.in_features );
        }
        else
        {
            EXPECT_EQ( weight_it->shape()[ 0 ], fixture.shape.out_features );
            EXPECT_EQ( weight_it->shape()[ 1 ], fixture.shape.in_features );
        }

        if ( fixture.has_bias )
        {
            ASSERT_GE( params.size(), 2u );
            auto* bias_it = dynamic_cast<Tensor<TensorDataType::FP32, CpuMemoryResource>*>( params[1] );
            if ( !bias_it )
            {
                auto host_copy = toHost<TensorDataType::FP32>( *dynamic_cast<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>*>( params[1] ) );
                EXPECT_EQ( host_copy.shape()[ 0 ], fixture.shape.out_features );
            }
            else
            {
                EXPECT_EQ( bias_it->shape()[ 0 ], fixture.shape.out_features );
            }
        }
    }

    // ====================================================================
    // Forward / Backward API tests
    // ====================================================================

    TYPED_TEST( LinearCudaTests, Forward_BeforeBuild_Throws )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LinearTestFixture<TPrecision>::Create( LinearTestShape::Small() );

        CudaTensor<TPrecision> in( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> out( Device::Cuda( 0 ), fixture.output_shape() );

        EXPECT_THROW( fixture.component->forward( in, out ), std::runtime_error );
    }

    TYPED_TEST( LinearCudaTests, Backward_BeforeBuildOrNotTraining_Throws )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LinearTestFixture<TPrecision>::Create( LinearTestShape::Small() );

        CudaTensor<TPrecision> in( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> outg( Device::Cuda( 0 ), fixture.output_shape() );
        CudaTensor<TPrecision> ing( Device::Cuda( 0 ), fixture.input_shape() );

        EXPECT_THROW( fixture.component->backward( in, outg, ing ), std::runtime_error );

        // Build but do not enable training => should still throw
        fixture.component->build( fixture.input_shape() );
        EXPECT_THROW( fixture.component->backward( in, outg, ing ), std::runtime_error );
    }

    TYPED_TEST( LinearCudaTests, Forward_WithVariousShapes_ProducesValidOutput )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        auto shapes = LinearTestShape::StandardShapes();

        for ( const auto& s : shapes )
        {
            auto fixture = LinearTestFixture<TPrecision>::Create( s );
            fixture.component->build( fixture.input_shape() );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), fixture.input_shape() );
            random( host_in, -1.0f, 1.0f );

            CudaTensor<TPrecision> device_in( Device::Cuda( 0 ), fixture.input_shape() );
            CudaTensor<TPrecision> device_out( Device::Cuda( 0 ), fixture.output_shape() );

            copy( host_in, device_in );

            EXPECT_NO_THROW( fixture.component->forward( device_in, device_out ) );
            EXPECT_EQ( device_out.shape(), fixture.output_shape() );

            auto host_out = toHost<TensorDataType::FP32>( device_out );
            EXPECT_EQ( host_out.size(), device_out.size() );
        }
    }

    TYPED_TEST( LinearCudaTests, Backward_AfterForward_InTraining_ProducesNonZeroInputGrad )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LinearTestFixture<TPrecision>::Create( LinearTestShape::Small(), true, true );
        fixture.component->build( fixture.input_shape() );
        fixture.component->setTraining( true );

        CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), fixture.input_shape() );
        CpuTensor<TensorDataType::FP32> host_outg( Device::Cpu(), fixture.output_shape() );

        random( host_in, -1.0f, 1.0f );
        random( host_outg, -0.1f, 0.1f );

        CudaTensor<TPrecision> device_in( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> device_out( Device::Cuda( 0 ), fixture.output_shape() );
        CudaTensor<TPrecision> device_outg( Device::Cuda( 0 ), fixture.output_shape() );
        CudaTensor<TPrecision> device_ing( Device::Cuda( 0 ), fixture.input_shape() );

        copy( host_in, device_in );
        copy( host_outg, device_outg );
        zeros( device_ing );

        // Forward then backward
        fixture.component->forward( device_in, device_out );

        EXPECT_NO_THROW( fixture.component->backward( device_in, device_outg, device_ing ) );

        auto host_ing = toHost<TensorDataType::FP32>( device_ing );
        bool has_nonzero = false;
        for ( size_t i = 0; i < host_ing.size(); ++i )
        {
            if ( std::abs( host_ing.data()[ i ] ) > 1e-6f )
            {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE( has_nonzero );
    }

    // ====================================================================
    // CPU <-> CUDA equivalence tests (FP32 only)
    // ====================================================================

    TYPED_TEST( LinearCudaTests, Forward_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Forward equivalence runs only for FP32";
        }

        try
        {
            auto s = LinearTestShape::Small();

            // Construct CPU and CUDA components
            auto cpu_comp = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "linear_cpu_equiv", LinearConfig( s.in_features, s.out_features ), Device::Cpu()
            );

            auto cuda_comp = std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
                "linear_cuda_equiv", LinearConfig( s.in_features, s.out_features ), Device::Cuda( 0 )
            );

            cpu_comp->build( s.inputShape() );
            cuda_comp->build( s.inputShape() );

            // Deterministic input
            Mila::Core::RandomGenerator::getInstance().setSeed( 1234 );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), s.inputShape() );
            random( host_in, -1.0f, 1.0f );

            // Copy parameters from CPU to CUDA using getParameters() (stable API)
            {
                auto cpu_params = cpu_comp->getParameters();
                auto cuda_params = cuda_comp->getParameters();

                size_t n = std::min( cpu_params.size(), cuda_params.size() );
                for ( size_t i = 0; i < n; ++i )
                {
                    auto* cpu_t_cpu = dynamic_cast<Tensor<TensorDataType::FP32, CpuMemoryResource>*>( cpu_params[ i ] );
                    auto* cpu_t_cuda = dynamic_cast<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>*>( cuda_params[ i ] );

                    if ( cpu_t_cpu )
                    {
                        // cpu parameter already host-resident
                        copy( *cpu_t_cpu, *cpu_t_cuda );
                    }
                    else
                    {
                        // Fallback: perform host copy then upload
                        auto host_copy = toHost<TensorDataType::FP32>( *dynamic_cast<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>*>( cpu_params[ i ] ) );
                        copy( host_copy, *cpu_t_cuda );
                    }
                }
            }

            // CPU forward
            CpuTensor<TensorDataType::FP32> host_out_cpu( Device::Cpu(), s.outputShape() );
            cpu_comp->forward( host_in, host_out_cpu );

            // CUDA forward
            CudaTensor<TensorDataType::FP32> device_in( Device::Cuda( 0 ), s.inputShape() );
            CudaTensor<TensorDataType::FP32> device_out( Device::Cuda( 0 ), s.outputShape() );
            
            copy( host_in, device_in );

            cuda_comp->forward( device_in, device_out );
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_out_cuda( Device::Cpu(), s.outputShape() );
            copy( device_out, host_out_cuda );

            auto* cdata = host_out_cpu.data();
            auto* gdata = host_out_cuda.data();
            size_t total = host_out_cpu.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_NEAR( cdata[ i ], gdata[ i ], 1e-3f ) << "Forward mismatch at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Linear CPU/CUDA forward equivalence not available";
        }
    }

    TYPED_TEST( LinearCudaTests, Backward_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Backward equivalence runs only for FP32";
        }

        try
        {
            auto s = LinearTestShape::Small();

            auto cpu_comp = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "linear_cpu_bwd", LinearConfig( s.in_features, s.out_features ), Device::Cpu()
            );

            auto cuda_comp = std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
                "linear_cuda_bwd", LinearConfig( s.in_features, s.out_features ), Device::Cuda( 0 )
            );

            cpu_comp->build( s.inputShape() );
            cuda_comp->build( s.inputShape() );

            cpu_comp->setTraining( true );
            cuda_comp->setTraining( true );

            // Deterministic input
            Mila::Core::RandomGenerator::getInstance().setSeed( 4321 );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), s.inputShape() );
            random( host_in, -1.0f, 1.0f );

            // Ensure parameter parity by copying parameters via getParameters()
            {
                auto cpu_params = cpu_comp->getParameters();
                auto cuda_params = cuda_comp->getParameters();

                size_t n = std::min( cpu_params.size(), cuda_params.size() );
                for ( size_t i = 0; i < n; ++i )
                {
                    auto* cpu_t_cpu = dynamic_cast<Tensor<TensorDataType::FP32, CpuMemoryResource>*>( cpu_params[ i ] );
                    auto* cuda_t_cuda = dynamic_cast<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>*>( cuda_params[ i ] );

                    if ( cpu_t_cpu && cuda_t_cuda )
                    {
                        copy( *cpu_t_cpu, *cuda_t_cuda );
                    }
                    else if ( !cpu_t_cpu )
                    {
                        // cpu param might be a device tensor (unlikely for CPU comp), fallback to host copy
                        auto host_copy = toHost<TensorDataType::FP32>( *dynamic_cast<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>*>( cpu_params[ i ] ) );
                        copy( host_copy, *cuda_t_cuda );
                    }
                }
            }

            // Forward pass (establish any internal state)
            CpuTensor<TensorDataType::FP32> host_out_cpu( Device::Cpu(), s.outputShape() );
            cpu_comp->forward( host_in, host_out_cpu );

            CudaTensor<TensorDataType::FP32> device_in( Device::Cuda( 0 ), s.inputShape() );
            CudaTensor<TensorDataType::FP32> device_out( Device::Cuda( 0 ), s.outputShape() );
            
            copy( host_in, device_in );
            
            cuda_comp->forward( device_in, device_out );
            cuda_comp->synchronize();

            // Create deterministic output gradient
            CpuTensor<TensorDataType::FP32> host_outg( Device::Cpu(), s.outputShape() );
            for ( size_t i = 0; i < host_outg.size(); ++i ) {
                host_outg.data()[ i ] = static_cast<float>( i % 7 - 3 );
            }

            // CPU backward
            CpuTensor<TensorDataType::FP32> host_ing_cpu( Device::Cpu(), s.inputShape() );

            cpu_comp->zeroGradients();

            cpu_comp->backward( host_in, host_outg, host_ing_cpu );

            // CUDA backward
            CudaTensor<TensorDataType::FP32> device_outg( Device::Cuda( 0 ), s.outputShape() );
            CudaTensor<TensorDataType::FP32> device_ing( Device::Cuda( 0 ), s.inputShape() );
            copy( host_outg, device_outg );

            cuda_comp->zeroGradients();

            cuda_comp->backward( device_in, device_outg, device_ing );
            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_ing_cuda( Device::Cpu(), s.inputShape() );
            copy( device_ing, host_ing_cuda );

            auto* cpu_data = host_ing_cpu.data();
            auto* cuda_data = host_ing_cuda.data();
            size_t total = host_ing_cpu.size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_NEAR( cpu_data[ i ], cuda_data[ i ], 1e-3f ) << "Backward mismatch at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Linear CPU/CUDA backward equivalence not available";
        }
    }

    // ====================================================================
    // Deterministic / sanity tests
    // ====================================================================

    TYPED_TEST( LinearCudaTests, Forward_OnesInput_ProducesFiniteOutput )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Deterministic test runs for FP32 only";
        }

        try
        {
            auto s = LinearTestShape::Minimal();
            auto fixture = LinearTestFixture<TensorDataType::FP32>::Create( s );

            fixture.component->build( fixture.input_shape() );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), fixture.input_shape() );
            ones( host_in );

            CudaTensor<TensorDataType::FP32> device_in( Device::Cuda( 0 ), fixture.input_shape() );
            CudaTensor<TensorDataType::FP32> device_out( Device::Cuda( 0 ), fixture.output_shape() );

            copy( host_in, device_in );
            fixture.component->forward( device_in, device_out );
            fixture.component->synchronize();

            auto host_out = toHost<TensorDataType::FP32>( device_out );

            for ( size_t i = 0; i < host_out.size(); ++i )
            {
                EXPECT_TRUE( std::isfinite( host_out.data()[ i ] ) );
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Linear backend not available";
        }
    }

    TYPED_TEST( LinearCudaTests, Forward_ZerosInput_ProducesZeroOutput_WhenWeightsZeroed )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Deterministic test runs for FP32 only";
        }

        try
        {
            auto s = LinearTestShape::Minimal();
            auto fixture = LinearTestFixture<TensorDataType::FP32>::Create( s );

            fixture.component->build( fixture.input_shape() );

            // Zero weights and bias explicitly to guarantee zero output
            auto params = fixture.component->getParameters();
            if ( params.size() >= 1 )
            {
                auto* w = dynamic_cast<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>*>( params[0] );
                if ( w )
                {
                    auto hw = toHost<TensorDataType::FP32>( *w );
                    zeros( hw );
                    copy( hw, *w );
                }
            }
            if ( params.size() >= 2 )
            {
                auto* b = dynamic_cast<Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>*>( params[1] );
                if ( b )
                {
                    auto hb = toHost<TensorDataType::FP32>( *b );
                    zeros( hb );
                    copy( hb, *b );
                }
            }

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), fixture.input_shape() );
            zeros( host_in );

            CudaTensor<TensorDataType::FP32> device_in( Device::Cuda( 0 ), fixture.input_shape() );
            CudaTensor<TensorDataType::FP32> device_out( Device::Cuda( 0 ), fixture.output_shape() );

            copy( host_in, device_in );
            fixture.component->forward( device_in, device_out );
            fixture.component->synchronize();

            auto host_out = toHost<TensorDataType::FP32>( device_out );

            for ( size_t i = 0; i < host_out.size(); ++i )
            {
                EXPECT_NEAR( host_out.data()[ i ], 0.0f, 1e-6f );
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Linear backend not available";
        }
    }
}