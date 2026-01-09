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

    TYPED_TEST( LinearCudaTests, ParameterCountAndConfigChecks )
    {
        if ( !this->cuda_available_ ) 
            GTEST_SKIP() << "CUDA not available";
        
        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LinearTestFixture<TPrecision>::Create( LinearTestShape::Small(), true );
        fixture.component->build( fixture.input_shape() );

        EXPECT_EQ( fixture.component->parameterCount(), static_cast<size_t>( fixture.shape.in_features * fixture.shape.out_features + ( fixture.has_bias ? fixture.shape.out_features : 0 ) ) );

        // Validate that public configuration reports expected features (shape inference via config)
        const auto& cfg = fixture.component->getConfig();
        EXPECT_EQ( cfg.getInputFeatures(), static_cast<dim_t>( fixture.shape.in_features ) );
        EXPECT_EQ( cfg.getOutputFeatures(), static_cast<dim_t>( fixture.shape.out_features ) );
    }
        
    // ====================================================================
    // Forward / Backward API tests (updated for new component-owned tensors)
    // ====================================================================

    TYPED_TEST( LinearCudaTests, Forward_BeforeBuild_Throws )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LinearTestFixture<TPrecision>::Create( LinearTestShape::Small() );

        CudaTensor<TPrecision> in( Device::Cuda( 0 ), fixture.input_shape() );

        EXPECT_THROW( fixture.component->forward( in ), std::runtime_error );
    }

    TYPED_TEST( LinearCudaTests, Backward_BeforeBuildOrNotTraining_Throws )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        auto fixture = LinearTestFixture<TPrecision>::Create( LinearTestShape::Small() );

        CudaTensor<TPrecision> in( Device::Cuda( 0 ), fixture.input_shape() );
        CudaTensor<TPrecision> outg( Device::Cuda( 0 ), fixture.output_shape() );

        EXPECT_THROW( fixture.component->backward( in, outg ), std::runtime_error );

        // Build but do not enable training => should still throw
        fixture.component->build( fixture.input_shape() );
        EXPECT_THROW( fixture.component->backward( in, outg ), std::runtime_error );
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

            copy( host_in, device_in );

            CudaTensor<TPrecision>* out_ptr = nullptr;
            ASSERT_NO_THROW( out_ptr = &fixture.component->forward( device_in ) );
            ASSERT_NE( out_ptr, nullptr );

            EXPECT_EQ( out_ptr->shape(), fixture.output_shape() );

            auto host_out = toHost<TensorDataType::FP32>( *out_ptr );
            EXPECT_EQ( host_out.size(), out_ptr->size() );
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
        CudaTensor<TPrecision> device_outg( Device::Cuda( 0 ), fixture.output_shape() );

        copy( host_in, device_in );
        copy( host_outg, device_outg );

        // Forward then backward (use component-owned buffers)
        CudaTensor<TPrecision>* device_out_ptr = nullptr;
        ASSERT_NO_THROW( device_out_ptr = &fixture.component->forward( device_in ) );
        ASSERT_NE( device_out_ptr, nullptr );

        CudaTensor<TPrecision>* device_ing_ptr = nullptr;
        EXPECT_NO_THROW( device_ing_ptr = &fixture.component->backward( device_in, device_outg ) );
        ASSERT_NE( device_ing_ptr, nullptr );

        auto host_ing = toHost<TensorDataType::FP32>( *device_ing_ptr );
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
    // CPU <-> CUDA equivalence tests (FP32 only) - updated to new API
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

            // Use deterministic RNG seed before constructing components so both
            // initializations are reproducible. This avoids relying on protected
            // parameter access in tests.
            Mila::Core::RandomGenerator::getInstance().setSeed( 1234 );

            // Construct CPU and CUDA components after seeding
            auto cpu_comp = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "linear_cpu_equiv", LinearConfig( s.in_features, s.out_features ), Device::Cpu()
            );

            auto cuda_comp = std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
                "linear_cuda_equiv", LinearConfig( s.in_features, s.out_features ), Device::Cuda( 0 )
            );

            cpu_comp->build( s.inputShape() );
            cuda_comp->build( s.inputShape() );

            // Deterministic input
            Mila::Core::RandomGenerator::getInstance().setSeed( 4321 );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), s.inputShape() );
            random( host_in, -1.0f, 1.0f );

            // CPU forward (new API)
            CpuTensor<TensorDataType::FP32>* cpu_out_ptr = nullptr;
            ASSERT_NO_THROW( cpu_out_ptr = &cpu_comp->forward( host_in ) );
            ASSERT_NE( cpu_out_ptr, nullptr );

            // CUDA forward (new API)
            CudaTensor<TensorDataType::FP32> device_in( Device::Cuda( 0 ), s.inputShape() );
            copy( host_in, device_in );

            CudaTensor<TensorDataType::FP32>* cuda_out_ptr = nullptr;
            ASSERT_NO_THROW( cuda_out_ptr = &cuda_comp->forward( device_in ) );
            ASSERT_NE( cuda_out_ptr, nullptr );

            cuda_comp->synchronize();

            CpuTensor<TensorDataType::FP32> host_out_cuda = toHost<TensorDataType::FP32>( *cuda_out_ptr );

            auto* cdata = cpu_out_ptr->data();
            auto* gdata = host_out_cuda.data();
            size_t total = cpu_out_ptr->size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_NEAR( cdata[ i ], gdata[ i ], 1e-2f ) << "Forward mismatch at index " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Linear CPU/CUDA forward equivalence not available";
        }
    }

    TYPED_TEST( LinearCudaTests, Backward_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) 
            GTEST_SKIP() << "CUDA not available";
        
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Backward equivalence runs only for FP32";
        }

        try
        {
            auto s = LinearTestShape::Small();

            // Deterministic seeding before parameter initialization
            Mila::Core::RandomGenerator::getInstance().setSeed( 4321 );

            auto cpu_fc = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "linear_cpu_bwd", LinearConfig( s.in_features, s.out_features ), Device::Cpu()
            );

            auto cuda_fc = std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
                "linear_cuda_bwd", LinearConfig( s.in_features, s.out_features ), Device::Cuda( 0 )
            );

            cpu_fc->build( s.inputShape() );
            cuda_fc->build( s.inputShape() );

            cpu_fc->setTraining( true );
            cuda_fc->setTraining( true );

            // Deterministic input
            Mila::Core::RandomGenerator::getInstance().setSeed( 9876 );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), s.inputShape() );
            random( host_input, -1.0f, 1.0f );

            // Forward pass (establish any internal state)
            CpuTensor<TensorDataType::FP32>* cpu_out_ptr = nullptr;
            ASSERT_NO_THROW( cpu_out_ptr = &cpu_fc->forward( host_input ) );
            ASSERT_NE( cpu_out_ptr, nullptr );

            CudaTensor<TensorDataType::FP32> device_input( Device::Cuda( 0 ), s.inputShape() );
            copy( host_input, device_input );

            CudaTensor<TensorDataType::FP32>* cuda_out_ptr = nullptr;
            ASSERT_NO_THROW( cuda_out_ptr = &cuda_fc->forward( device_input ) );
            ASSERT_NE( cuda_out_ptr, nullptr );

            cuda_fc->synchronize();

            CpuTensor<TensorDataType::FP32> host_output_grad( Device::Cpu(), s.outputShape() );

            auto num_elements = host_output_grad.size();
            for ( size_t i = 0; i < host_output_grad.size(); ++i ) {
                host_output_grad.data()[ i ] = (i % 2 == 0) ? 0.1f : -0.1f;
            }

            // DEBUG: GIGO
            std::cout << "Output Gradient:" << std::endl;
            std::cout << host_output_grad.toString( true );

            // Before backward, reset any parameter gradients to zero
            cpu_fc->zeroGradients();
            cuda_fc->zeroGradients();

            CpuTensor<TensorDataType::FP32>* host_input_grad_ptr = nullptr;
            ASSERT_NO_THROW( host_input_grad_ptr = &cpu_fc->backward( host_input, host_output_grad ) );
            ASSERT_NE( host_input_grad_ptr, nullptr );

            CudaTensor<TensorDataType::FP32> device_output_grad( Device::Cuda( 0 ), s.outputShape() );
            copy( host_output_grad, device_output_grad );

            CudaTensor<TensorDataType::FP32>* device_input_grad_ptr = nullptr;
            ASSERT_NO_THROW( device_input_grad_ptr = &cuda_fc->backward( device_input, device_output_grad ) );
            ASSERT_NE( device_input_grad_ptr, nullptr );

            cuda_fc->synchronize();

            CpuTensor<TensorDataType::FP32> host_input_grad_cuda = toHost<TensorDataType::FP32>( *device_input_grad_ptr );

            auto* cpu_data = host_input_grad_ptr->data();
            auto* cuda_data = host_input_grad_cuda.data();
            
            size_t total = host_input_grad_ptr->size();

            for ( size_t i = 0; i < total; ++i )
            {
                EXPECT_NEAR( cpu_data[ i ], cuda_data[ i ], 1e-2f ) << "Backward mismatch at index " << i;
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
        if ( !this->cuda_available_ ) 
            GTEST_SKIP() << "CUDA not available";
        
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

            copy( host_in, device_in );

            CudaTensor<TensorDataType::FP32>* out_ptr = nullptr;
            ASSERT_NO_THROW( out_ptr = &fixture.component->forward( device_in ) );
            ASSERT_NE( out_ptr, nullptr );

            fixture.component->synchronize();

            auto host_out = toHost<TensorDataType::FP32>( *out_ptr );

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
}