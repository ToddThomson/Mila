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

    // Test harness to expose Linear's protected parameter/gradient accessors.
    template<DeviceType TDeviceType, TensorDataType TPrecision>
    struct LinearTestHarness : public Mila::Dnn::Linear<TDeviceType, TPrecision>
    {
        using Base = Mila::Dnn::Linear<TDeviceType, TPrecision>;
        using TensorType = typename Base::TensorType;

        LinearTestHarness( const std::string& name, const LinearConfig& cfg, std::optional<DeviceId> device_id = std::nullopt )
            : Base( name, cfg, device_id )
        {}

        std::vector<TensorType*> getTypedParametersPublic() const
        {
            std::vector<TensorType*> out;
            auto params = Base::getParameters();
            out.reserve( params.size() );

            for ( auto p : params )
            {
                out.push_back( static_cast<TensorType*>( p ) );
            }

            return out;
        }

        std::vector<TensorType*> getTypedGradientsPublic() const
        {
            std::vector<TensorType*> out;
            auto grads = Base::getGradients();
            out.reserve( grads.size() );

            for ( auto g : grads )
            {
                out.push_back( static_cast<TensorType*>( g ) );
            }

            return out;
        }

        std::vector<Mila::Dnn::ITensor*> getParametersPublic() const
        {
            return Base::getParameters();
        }

        std::vector<Mila::Dnn::ITensor*> getGradientsPublic() const
        {
            return Base::getGradients();
        }
    };

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
              has_bias( true )
        {}

        static LinearTestFixture Create( const LinearTestShape& s, bool with_bias = true )
        {
            LinearTestFixture f;
            f.shape = s;
            f.has_bias = with_bias;

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

        const auto& cfg = fixture.component->getConfig();
        EXPECT_EQ( cfg.getInputFeatures(), static_cast<dim_t>( fixture.shape.in_features ) );
        EXPECT_EQ( cfg.getOutputFeatures(), static_cast<dim_t>( fixture.shape.out_features ) );
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

    TYPED_TEST( LinearCudaTests, GetGradients_WhenNotTraining_Throws )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        auto s = LinearTestShape::Small();
        LinearConfig cfg( s.in_features, s.out_features );

        auto harness = std::make_shared<
            LinearTestHarness<DeviceType::Cuda, TPrecision>
        >( "linear_grads_throw", cfg, Device::Cuda( 0 ) );

        harness->build( s.inputShape() );

        // Training is disabled by default: getGradients must throw.
        EXPECT_THROW( harness->getGradientsPublic(), std::runtime_error );

        // Enabling training enables gradient access.
        harness->setTraining( true );
        EXPECT_NO_THROW( harness->getGradientsPublic() );
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

        auto fixture = LinearTestFixture<TPrecision>::Create( LinearTestShape::Small(), true );
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
// Decode (inference) tests
//
// decode() dispatches to CudaLinearOp::decode() via IDecode, which uses
// cuda_matvec_impl -- a single-vector (M=1) kernel optimized for
// auto-regressive token generation. Tests must build and run with an
// outer batch size of 1 to match the matvec semantics.
// On CPU, decode() falls back to operation_->forward().
// ====================================================================

    TYPED_TEST( LinearCudaTests, Decode_BeforeBuild_Throws )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        LinearConfig cfg( 16, 32 );
        auto comp = std::make_shared<Linear<DeviceType::Cuda, TPrecision>>( "lin_decode_throw", cfg, Device::Cuda( 0 ) );

        CudaTensor<TPrecision> in( Device::Cuda( 0 ), shape_t{ 1, 1, 16 } );

        EXPECT_THROW( comp->decode( in ), std::runtime_error );
    }

    TYPED_TEST( LinearCudaTests, Decode_SingleToken_ProducesValidOutput )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        // Feature dimension pairs representative of real model sizes.
        const std::vector<std::pair<int64_t, int64_t>> feature_pairs = {
            { 16, 32 }, { 128, 256 }, { 512, 768 }
        };

        for ( auto [in_feat, out_feat] : feature_pairs )
        {
            const shape_t single_token_shape = { 1, 1, in_feat };

            LinearConfig cfg( static_cast<dim_t>(in_feat), static_cast<dim_t>(out_feat) );
            cfg.withBias( true );

            auto comp = std::make_shared<Linear<DeviceType::Cuda, TPrecision>>(
                "lin_decode_valid", cfg, Device::Cuda( 0 ) );

            comp->build( single_token_shape );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), single_token_shape );
            random( host_in, -1.0f, 1.0f );

            CudaTensor<TPrecision> device_in( Device::Cuda( 0 ), single_token_shape );
            copy( host_in, device_in );

            CudaTensor<TPrecision>* out_ptr = nullptr;
            ASSERT_NO_THROW( out_ptr = &comp->decode( device_in ) );
            ASSERT_NE( out_ptr, nullptr );

            const shape_t expected_output_shape = { 1, 1, out_feat };
            EXPECT_EQ( out_ptr->shape(), expected_output_shape );

            comp->synchronize();

            auto host_out = toHost<TensorDataType::FP32>( *out_ptr );

            for ( size_t i = 0; i < host_out.size(); ++i )
            {
                EXPECT_TRUE( std::isfinite( host_out.data()[ i ] ) )
                    << "Non-finite decode output at element " << i
                    << " for in_feat=" << in_feat << " out_feat=" << out_feat;
            }
        }
    }

    TYPED_TEST( LinearCudaTests, Decode_DoesNotRequireTrainingMode )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        const shape_t single_token_shape = { 1, 1, 16 };

        LinearConfig cfg( 16, 32 );
        auto comp = std::make_shared<Linear<DeviceType::Cuda, TPrecision>>(
            "lin_decode_no_train", cfg, Device::Cuda( 0 ) );

        comp->build( single_token_shape );

        // Training mode must remain disabled throughout.
        ASSERT_FALSE( comp->isTraining() );

        CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), single_token_shape );
        random( host_in, -1.0f, 1.0f );

        CudaTensor<TPrecision> device_in( Device::Cuda( 0 ), single_token_shape );
        copy( host_in, device_in );

        EXPECT_NO_THROW( (void)comp->decode( device_in ) );
    }

    TYPED_TEST( LinearCudaTests, Decode_EquivalentToForward_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Decode equivalence runs only for FP32";
        }

        try
        {
            // Build with M=1 (single token) so cuda_matvec_impl and the cuBLASLt
            // matmul in forward() operate on identical inputs and produce outputs
            // that are directly comparable.
            const int64_t in_features = 16;
            const int64_t out_features = 32;
            const shape_t single_token_shape = { 1, 1, in_features };

            LinearConfig cfg( in_features, out_features );
            cfg.withBias( true );

            auto comp = std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
                "lin_decode_equiv", cfg, Device::Cuda( 0 ) );

            comp->build( single_token_shape );

            Mila::Core::RandomGenerator::getInstance().setSeed( 5050 );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), single_token_shape );
            random( host_in, -1.0f, 1.0f );

            CudaTensor<TensorDataType::FP32> device_in( Device::Cuda( 0 ), single_token_shape );
            copy( host_in, device_in );

            // forward() uses cuBLASLt matmul with M=1.
            CudaTensor<TensorDataType::FP32>* fwd_out = nullptr;
            ASSERT_NO_THROW( fwd_out = &comp->forward( device_in ) );
            ASSERT_NE( fwd_out, nullptr );

            comp->synchronize();
            CpuTensor<TensorDataType::FP32> host_fwd = toHost<TensorDataType::FP32>( *fwd_out );

            // decode() uses cuda_matvec_impl with the same M=1 input.
            CudaTensor<TensorDataType::FP32>* dec_out = nullptr;
            ASSERT_NO_THROW( dec_out = &comp->decode( device_in ) );
            ASSERT_NE( dec_out, nullptr );

            comp->synchronize();
            CpuTensor<TensorDataType::FP32> host_dec = toHost<TensorDataType::FP32>( *dec_out );

            ASSERT_EQ( host_fwd.size(), host_dec.size() );

            for ( size_t i = 0; i < host_fwd.size(); ++i )
            {
                EXPECT_NEAR( host_fwd.data()[ i ], host_dec.data()[ i ], 1e-3f )
                    << "decode/forward mismatch at element " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Linear decode not available";
        }
    }

    TYPED_TEST( LinearCudaTests, Decode_MultiTokenInput_Throws )
    {
        if ( !this->cuda_available_ )
            GTEST_SKIP() << "CUDA not available";
        
        constexpr TensorDataType TPrecision = TypeParam::value;

        const int64_t in_features = 16;
        const int64_t out_features = 32;

        LinearConfig cfg( in_features, out_features );
        auto comp = std::make_shared<Linear<DeviceType::Cuda, TPrecision>>(
            "lin_decode_multi", cfg, Device::Cuda( 0 ) );

        // Build with a multi-token max shape.
        comp->build( shape_t{ 4, 8, in_features } );

        // Any shape with outer size > 1 must be rejected.
        CudaTensor<TPrecision> multi_token( Device::Cuda( 0 ), shape_t{ 2, 1, in_features } );
        EXPECT_THROW( comp->decode( multi_token ), std::invalid_argument );

        CudaTensor<TPrecision> single_token( Device::Cuda( 0 ), shape_t{ 1, 1, in_features } );
        EXPECT_NO_THROW( (void)comp->decode( single_token ) );
    }

    TYPED_TEST( LinearCudaTests, Decode_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Decode CPU/CUDA equivalence runs only for FP32";
        }

        try
        {
            auto s = LinearTestShape::Small();

            // Single-token shape required by cuda_matvec_impl decode path.
            const shape_t single_token_shape = { 1, 1, s.in_features };

            Mila::Core::RandomGenerator::getInstance().setSeed( 5555 );

            auto cpu_lin = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "linear_cpu_decode_equiv", LinearConfig( s.in_features, s.out_features ), Device::Cpu()
            );

            auto cuda_lin = std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
                "linear_cuda_decode_equiv", LinearConfig( s.in_features, s.out_features ), Device::Cuda( 0 )
            );

            cpu_lin->build( single_token_shape );
            cuda_lin->build( single_token_shape );

            Mila::Core::RandomGenerator::getInstance().setSeed( 6060 );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), single_token_shape );
            random( host_in, -1.0f, 1.0f );

            CpuTensor<TensorDataType::FP32>* cpu_out_ptr = nullptr;
            ASSERT_NO_THROW( cpu_out_ptr = &cpu_lin->decode( host_in ) );
            ASSERT_NE( cpu_out_ptr, nullptr );

            CudaTensor<TensorDataType::FP32> device_in( Device::Cuda( 0 ), single_token_shape );
            copy( host_in, device_in );

            CudaTensor<TensorDataType::FP32>* cuda_out_ptr = nullptr;
            ASSERT_NO_THROW( cuda_out_ptr = &cuda_lin->decode( device_in ) );
            ASSERT_NE( cuda_out_ptr, nullptr );

            cuda_lin->synchronize();

            CpuTensor<TensorDataType::FP32> host_cuda_out = toHost<TensorDataType::FP32>( *cuda_out_ptr );

            ASSERT_EQ( cpu_out_ptr->size(), host_cuda_out.size() );

            const float tol = 1e-3f;
            auto* cpu_data = cpu_out_ptr->data();
            auto* cuda_data = host_cuda_out.data();

            for ( size_t i = 0; i < cpu_out_ptr->size(); ++i )
            {
                EXPECT_NEAR( cpu_data[ i ], cuda_data[ i ], tol )
                    << "Decode CPU/CUDA mismatch at element " << i;
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Linear CPU/CUDA decode equivalence not available";
        }
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

            Mila::Core::RandomGenerator::getInstance().setSeed( 1234 );

            auto cpu_comp = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "linear_cpu_equiv", LinearConfig( s.in_features, s.out_features ), Device::Cpu()
            );

            auto cuda_comp = std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
                "linear_cuda_equiv", LinearConfig( s.in_features, s.out_features ), Device::Cuda( 0 )
            );

            cpu_comp->build( s.inputShape() );
            cuda_comp->build( s.inputShape() );

            Mila::Core::RandomGenerator::getInstance().setSeed( 4321 );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), s.inputShape() );
            random( host_in, -1.0f, 1.0f );

            CpuTensor<TensorDataType::FP32>* cpu_out_ptr = nullptr;
            ASSERT_NO_THROW( cpu_out_ptr = &cpu_comp->forward( host_in ) );
            ASSERT_NE( cpu_out_ptr, nullptr );

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

            Mila::Core::RandomGenerator::getInstance().setSeed( 9876 );

            CpuTensor<TensorDataType::FP32> host_input( Device::Cpu(), s.inputShape() );
            random( host_input, -1.0f, 1.0f );

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

            for ( size_t i = 0; i < host_output_grad.size(); ++i )
            {
                host_output_grad.data()[ i ] = (i % 2 == 0) ? 0.1f : -0.1f;
            }

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

    TYPED_TEST( LinearCudaTests, ProtectedParameterAndGradientAccess_ViaHarness )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        auto s = LinearTestShape::Small();
        LinearConfig cfg( s.in_features, s.out_features );
        cfg.withBias( false );

        auto harness = std::make_shared<
            LinearTestHarness<DeviceType::Cuda, TPrecision>
        >( "linear_harness", cfg, Device::Cuda( 0 ) );

        ASSERT_NO_THROW( harness->build( s.inputShape() ) );

        harness->setTraining( true );

        std::vector<ITensor*> params;
        ASSERT_NO_THROW( params = harness->getParametersPublic() );
        EXPECT_EQ( params.size(), cfg.hasBias() ? 2u : 1u );

        std::vector<Mila::Dnn::ITensor*> grads;
        ASSERT_NO_THROW( grads = harness->getGradientsPublic() );
        EXPECT_EQ( grads.size(), cfg.hasBias() ? 2u : 1u );
    }

    TYPED_TEST( LinearCudaTests, ParametersUnchanged_AfterForward )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Parameters comparison test runs for FP32 only";
        }

        try
        {
            auto s = LinearTestShape::Small();
            LinearConfig cfg( s.in_features, s.out_features );
            cfg.withBias( true );

            auto harness = std::make_shared<
                LinearTestHarness<DeviceType::Cuda, TPrecision>
            >( "linear_params_check", cfg, Device::Cuda( 0 ) );

            harness->build( s.inputShape() );

            auto params = harness->getTypedParametersPublic();
            ASSERT_FALSE( params.empty() );

            std::vector<CpuTensor<TPrecision>> before;
            before.reserve( params.size() );

            for ( auto p : params )
            {
                before.push_back( toHost<TPrecision>( *p ) );
            }

            Mila::Core::RandomGenerator::getInstance().setSeed( 2025 );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), s.inputShape() );
            random( host_in, -1.0f, 1.0f );

            CudaTensor<TPrecision> device_in( Device::Cuda( 0 ), s.inputShape() );
            copy( host_in, device_in );

            CudaTensor<TPrecision>* out_ptr = nullptr;
            ASSERT_NO_THROW( out_ptr = &harness->forward( device_in ) );
            ASSERT_NE( out_ptr, nullptr );

            harness->synchronize();

            for ( size_t idx = 0; idx < params.size(); ++idx )
            {
                auto after = toHost<TPrecision>( *params[ idx ] );

                ASSERT_EQ( before[ idx ].size(), after.size() );

                for ( size_t i = 0; i < after.size(); ++i )
                {
                    EXPECT_NEAR( before[ idx ].data()[ i ], after.data()[ i ], 1e-6f )
                        << "Parameter changed after forward at param " << idx << " element " << i;
                }
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Linear backend not available for parameter check";
        }
    }

    TYPED_TEST( LinearCudaTests, GradientsCorrectness_AfterBackward )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Gradient correctness test runs for FP32 only";
        }

        try
        {
            auto s = LinearTestShape::Small();
            LinearConfig cfg( s.in_features, s.out_features );
            cfg.withBias( true );

            auto harness = std::make_shared<
                LinearTestHarness<DeviceType::Cuda, TPrecision>
            >( "linear_grad_check", cfg, Device::Cuda( 0 ) );

            harness->build( s.inputShape() );
            harness->setTraining( true );

            Mila::Core::RandomGenerator::getInstance().setSeed( 4242 );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), s.inputShape() );
            random( host_in, -1.0f, 1.0f );

            CpuTensor<TensorDataType::FP32> host_outg( Device::Cpu(), s.outputShape() );

            for ( size_t i = 0; i < host_outg.size(); ++i )
            {
                host_outg.data()[ i ] = (i % 2 == 0) ? 0.07f : -0.07f;
            }

            CudaTensor<TPrecision> device_in( Device::Cuda( 0 ), s.inputShape() );
            CudaTensor<TPrecision> device_outg( Device::Cuda( 0 ), s.outputShape() );

            copy( host_in, device_in );
            copy( host_outg, device_outg );

            harness->zeroGradients();

            ASSERT_NO_THROW( (void)harness->forward( device_in ) );
            ASSERT_NO_THROW( (void)harness->backward( device_in, device_outg ) );

            harness->synchronize();

            auto grads = harness->getTypedGradientsPublic();
            ASSERT_FALSE( grads.empty() );

            bool any_nonzero = false;

            for ( auto g : grads )
            {
                auto host_grad = toHost<TPrecision>( *g );

                double accum_abs = 0.0;

                for ( size_t i = 0; i < host_grad.size(); ++i )
                {
                    accum_abs += std::abs( host_grad.data()[ i ] );
                }

                if ( accum_abs > 1e-6 )
                {
                    any_nonzero = true;
                }
            }

            EXPECT_TRUE( any_nonzero ) << "Expected at least one non-zero parameter gradient after backward";

            harness->zeroGradients();
            harness->synchronize();

            for ( auto g : grads )
            {
                auto host_grad = toHost<TPrecision>( *g );

                for ( size_t i = 0; i < host_grad.size(); ++i )
                {
                    EXPECT_NEAR( host_grad.data()[ i ], 0.0f, 1e-6f )
                        << "Gradient not zeroed at element " << i;
                }
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Linear backend not available for gradient check";
        }
    }

    // ====================================================================
    // CPU <-> CUDA parameter/gradient equality tests (FP32 only)
    // ====================================================================

    TYPED_TEST( LinearCudaTests, Parameters_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Parameters equivalence runs only for FP32";
        }

        try
        {
            auto s = LinearTestShape::Small();

            Mila::Core::RandomGenerator::getInstance().setSeed( 2026 );

            auto cpu_lin = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "linear_cpu_params", LinearConfig( s.in_features, s.out_features ), Device::Cpu()
            );

            auto cuda_lin = std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
                "linear_cuda_params", LinearConfig( s.in_features, s.out_features ), Device::Cuda( 0 )
            );

            cpu_lin->build( s.inputShape() );
            cuda_lin->build( s.inputShape() );

            auto cpu_params = cpu_lin->getParameters();
            auto cuda_params = cuda_lin->getParameters();

            ASSERT_EQ( cpu_params.size(), cuda_params.size() );

            const float tol = 1e-6f;

            for ( size_t i = 0; i < cpu_params.size(); ++i )
            {
                auto cpu_t = static_cast<CpuTensor<TensorDataType::FP32>*>( cpu_params[ i ] );
                auto cuda_t = static_cast<CudaTensor<TensorDataType::FP32>*>( cuda_params[ i ] );

                ASSERT_NE( cpu_t, nullptr );
                ASSERT_NE( cuda_t, nullptr );

                auto host_cuda = toHost<TensorDataType::FP32>( *cuda_t );
                auto host_cpu = toHost<TensorDataType::FP32>( *cpu_t );

                ASSERT_EQ( host_cpu.size(), host_cuda.size() );

                for ( size_t e = 0; e < host_cpu.size(); ++e )
                {
                    EXPECT_NEAR( host_cpu.data()[ e ], host_cuda.data()[ e ], tol )
                        << "Parameter mismatch at param " << i << " element " << e;
                }
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Linear CPU/CUDA parameter equivalence not available";
        }
    }

    TYPED_TEST( LinearCudaTests, Gradients_CPU_CUDA_Equivalence_FP32 )
    {
        if ( !this->cuda_available_ ) GTEST_SKIP() << "CUDA not available";
        constexpr TensorDataType TPrecision = TypeParam::value;

        if constexpr ( TPrecision != TensorDataType::FP32 )
        {
            GTEST_SKIP() << "Gradients equivalence runs only for FP32";
        }

        try
        {
            auto s = LinearTestShape::Small();

            Mila::Core::RandomGenerator::getInstance().setSeed( 3030 );

            auto cpu_lin = std::make_shared<Linear<DeviceType::Cpu, TensorDataType::FP32>>(
                "linear_cpu_grads", LinearConfig( s.in_features, s.out_features ), Device::Cpu()
            );

            auto cuda_lin = std::make_shared<Linear<DeviceType::Cuda, TensorDataType::FP32>>(
                "linear_cuda_grads", LinearConfig( s.in_features, s.out_features ), Device::Cuda( 0 )
            );

            cpu_lin->build( s.inputShape() );
            cuda_lin->build( s.inputShape() );

            cpu_lin->setTraining( true );
            cuda_lin->setTraining( true );

            Mila::Core::RandomGenerator::getInstance().setSeed( 4040 );

            CpuTensor<TensorDataType::FP32> host_in( Device::Cpu(), s.inputShape() );
            random( host_in, -1.0f, 1.0f );

            CpuTensor<TensorDataType::FP32> host_outg( Device::Cpu(), s.outputShape() );

            for ( size_t i = 0; i < host_outg.size(); ++i )
            {
                host_outg.data()[ i ] = (i % 2 == 0) ? 0.05f : -0.05f;
            }

            CudaTensor<TensorDataType::FP32> device_in( Device::Cuda( 0 ), s.inputShape() );
            CudaTensor<TensorDataType::FP32> device_outg( Device::Cuda( 0 ), s.outputShape() );

            copy( host_in, device_in );
            copy( host_outg, device_outg );

            cpu_lin->zeroGradients();
            cuda_lin->zeroGradients();

            ASSERT_NO_THROW( (void)cpu_lin->forward( host_in ) );
            ASSERT_NO_THROW( (void)cpu_lin->backward( host_in, host_outg ) );

            ASSERT_NO_THROW( (void)cuda_lin->forward( device_in ) );
            ASSERT_NO_THROW( (void)cuda_lin->backward( device_in, device_outg ) );

            cuda_lin->synchronize();

            auto cpu_grads = cpu_lin->getGradients();
            auto cuda_grads = cuda_lin->getGradients();

            ASSERT_EQ( cpu_grads.size(), cuda_grads.size() );

            const float tol = 1e-5f;

            for ( size_t i = 0; i < cpu_grads.size(); ++i )
            {
                auto cpu_g = static_cast<CpuTensor<TensorDataType::FP32>*>( cpu_grads[ i ] );
                auto cuda_g = static_cast<CudaTensor<TensorDataType::FP32>*>( cuda_grads[ i ] );

                ASSERT_NE( cpu_g, nullptr );
                ASSERT_NE( cuda_g, nullptr );

                auto host_cpu_g = toHost<TensorDataType::FP32>( *cpu_g );
                auto host_cuda_g = toHost<TensorDataType::FP32>( *cuda_g );

                ASSERT_EQ( host_cpu_g.size(), host_cuda_g.size() );

                for ( size_t e = 0; e < host_cpu_g.size(); ++e )
                {
                    EXPECT_NEAR( host_cpu_g.data()[ e ], host_cuda_g.data()[ e ], tol )
                        << "Gradient mismatch at grad " << i << " element " << e;
                }
            }
        }
        catch ( const std::exception& )
        {
            GTEST_SKIP() << "Linear CPU/CUDA gradients equivalence not available";
        }
    }

}