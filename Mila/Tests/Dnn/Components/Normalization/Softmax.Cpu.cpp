#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cstdint>
#include <stdexcept>

import Mila;

namespace Components_Normalization_Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    // ====================================================================
    // Test Shape Definitions
    // ====================================================================

    enum class TestShapeSize
    {
        Small,
        Medium,
        Large,
        Minimal
    };

    struct TestShape
    {
        TestShapeSize size;
        shape_t dimensions;
        std::string name;

        static TestShape Small()
        {
            return { TestShapeSize::Small, { 2, 3, 4 }, "Small" };
        }

        static TestShape Medium()
        {
            return { TestShapeSize::Medium, { 4, 128, 1024 }, "Medium" };
        }

        static TestShape Large()
        {
            return { TestShapeSize::Large, { 8, 256, 2048 }, "Large" };
        }

        static TestShape Minimal()
        {
            return { TestShapeSize::Minimal, { 1, 1, 8 }, "Minimal" };
        }

        static TestShape FromSize( TestShapeSize size )
        {
            switch ( size )
            {
                case TestShapeSize::Small:
                    return Small();
                case TestShapeSize::Medium:
                    return Medium();
                case TestShapeSize::Large:
                    return Large();
                case TestShapeSize::Minimal:
                    return Minimal();
                default:
                    return Small();
            }
        }
    };

    // ====================================================================
    // Test Fixture Structure
    // ====================================================================

    struct SoftmaxTestFixture
    {
        TestShape test_shape;
        SoftmaxConfig config;
        std::shared_ptr<Softmax<DeviceType::Cpu, TensorDataType::FP32>> component;
        int64_t axis;
        bool is_training;

        static SoftmaxTestFixture Create(
            TestShape shape,
            int64_t axis = -1,
            bool is_training = false )
        {
            SoftmaxTestFixture fixture;
            fixture.test_shape = shape;
            fixture.axis = axis;
            fixture.is_training = is_training;

            fixture.config = SoftmaxConfig();
            fixture.config.withAxis( axis );

            // Generate a stable valid name for the component (must start with a letter)
            std::string name = std::string( "softmax_" ) + shape.name;

            // Construct in standalone mode so component owns its ExecutionContext
            fixture.component = std::make_shared<Softmax<DeviceType::Cpu, TensorDataType::FP32>>(
                name,
                fixture.config,
                Device::Cpu() );

            fixture.component->setTraining( is_training );

            return fixture;
        }

        const shape_t& shape() const
        {
            return test_shape.dimensions;
        }
    };

    // ====================================================================
    // Test Fixture Class
    // ====================================================================

    class SoftmaxCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            default_axis_ = -1;
        }

        void TearDown() override
        {
            inference_fixtures_.clear();
            training_fixtures_.clear();
        }

        SoftmaxTestFixture& GetInferenceFixture( TestShapeSize size )
        {
            auto it = inference_fixtures_.find( size );

            if ( it == inference_fixtures_.end() )
            {
                TestShape shape = TestShape::FromSize( size );
                auto fixture = SoftmaxTestFixture::Create( shape, default_axis_, false );
                auto result = inference_fixtures_.emplace( size, std::move( fixture ) );
                it = result.first;
            }

            return it->second;
        }

        SoftmaxTestFixture& GetTrainingFixture( TestShapeSize size )
        {
            auto it = training_fixtures_.find( size );

            if ( it == training_fixtures_.end() )
            {
                TestShape shape = TestShape::FromSize( size );
                auto fixture = SoftmaxTestFixture::Create( shape, default_axis_, true );
                auto result = training_fixtures_.emplace( size, std::move( fixture ) );
                it = result.first;
            }

            return it->second;
        }

        int64_t default_axis_;
        std::map<TestShapeSize, SoftmaxTestFixture> inference_fixtures_;
        std::map<TestShapeSize, SoftmaxTestFixture> training_fixtures_;
    };

    // ====================================================================
    // Helper Functions
    // ====================================================================

    void ValidateNormalization( const CpuTensor<TensorDataType::FP32>& output, int64_t axis )
    {
        const auto& shape = output.shape();
        const int64_t ndim = static_cast<int64_t>(shape.size());

        int64_t normalized_axis = axis;
        if ( normalized_axis < 0 )
        {
            normalized_axis = ndim + normalized_axis;
        }

        int64_t outer_size = 1;
        for ( int64_t i = 0; i < normalized_axis; ++i )
        {
            outer_size *= shape[ i ];
        }

        int64_t dim_size = shape[ normalized_axis ];

        int64_t inner_size = 1;
        for ( int64_t i = normalized_axis + 1; i < ndim; ++i )
        {
            inner_size *= shape[ i ];
        }

        auto output_ptr = output.data();

        for ( int64_t outer = 0; outer < outer_size; ++outer )
        {
            for ( int64_t inner = 0; inner < inner_size; ++inner )
            {
                float sum = 0.0f;

                for ( int64_t i = 0; i < dim_size; ++i )
                {
                    size_t idx = (outer * dim_size * inner_size) + (i * inner_size) + inner;
                    sum += static_cast<float>( output_ptr[ idx ] );
                }

                EXPECT_NEAR( sum, 1.0f, 1e-4f )
                    << "Softmax sum not normalized at outer=" << outer << ", inner=" << inner;
            }
        }
    }

    // ====================================================================
    // Construction Tests
    // ====================================================================

    TEST_F( SoftmaxCpuTests, Constructor_WithValidDeviceId_CreatesComponent )
    {
        SoftmaxConfig config;
        config.withAxis( -1 );

        std::shared_ptr<Softmax<DeviceType::Cpu, TensorDataType::FP32>> component;

        ASSERT_NO_THROW(
            ( component = std::make_shared<Softmax<DeviceType::Cpu, TensorDataType::FP32>>(
                "ctor_device_cpu",
                config,
                Device::Cpu()
            ))
        );

        ASSERT_NE( component, nullptr );
        EXPECT_EQ( component->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( SoftmaxCpuTests, Constructor_WithoutDeviceId_CreatesComponent )
    {
        SoftmaxConfig config;
        config.withAxis( -1 );

        std::shared_ptr<Softmax<DeviceType::Cpu, TensorDataType::FP32>> component;

        ASSERT_NO_THROW(
            ( component = std::make_shared<Softmax<DeviceType::Cpu, TensorDataType::FP32>>(
                "ctor_shared_cpu",
                config ) )
        );

        ASSERT_NE( component, nullptr );
    }

    // ====================================================================
    // Device Type Tests
    // ====================================================================

    TEST_F( SoftmaxCpuTests, GetDeviceType_AfterConstruction_ReturnsCpu )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Small );

        EXPECT_EQ( fixture.component->getDeviceType(), DeviceType::Cpu );

        auto device = fixture.component->getDeviceId();

        EXPECT_EQ( device.type, DeviceType::Cpu );
    }

    // ====================================================================
    // Training Mode Tests
    // ====================================================================

    TEST_F( SoftmaxCpuTests, IsTraining_InferenceFixture_ReturnsFalse )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Small );

        EXPECT_FALSE( fixture.component->isTraining() );
    }

    TEST_F( SoftmaxCpuTests, IsTraining_TrainingFixture_ReturnsTrue )
    {
        auto& fixture = GetTrainingFixture( TestShapeSize::Medium );

        EXPECT_TRUE( fixture.component->isTraining() );
    }

    TEST_F( SoftmaxCpuTests, SetTraining_TogglingMode_UpdatesState )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Small );

        EXPECT_FALSE( fixture.component->isTraining() );

        fixture.component->setTraining( true );
        EXPECT_TRUE( fixture.component->isTraining() );

        fixture.component->setTraining( false );
        EXPECT_FALSE( fixture.component->isTraining() );
    }

    // ====================================================================
    // Build State Tests
    // ====================================================================

    TEST_F( SoftmaxCpuTests, IsBuilt_BeforeBuild_ReturnsFalse )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Small );

        EXPECT_FALSE( fixture.component->isBuilt() );
    }

    TEST_F( SoftmaxCpuTests, Build_WithSmallShape_SetsBuiltState )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Small );

        EXPECT_NO_THROW( fixture.component->build( fixture.shape() ) );
        EXPECT_TRUE( fixture.component->isBuilt() );
    }

    TEST_F( SoftmaxCpuTests, IsBuilt_AfterBuild_ReturnsTrue )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Small );

        fixture.component->build( fixture.shape() );

        EXPECT_TRUE( fixture.component->isBuilt() );
    }

    // ====================================================================
    // Parameter Count Tests
    // ====================================================================

    TEST_F( SoftmaxCpuTests, ParameterCount_AfterConstruction_ReturnsZero )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Small );

        EXPECT_EQ( fixture.component->parameterCount(), 0 );
    }

    // ====================================================================
    // String Representation Tests
    // ====================================================================

    TEST_F( SoftmaxCpuTests, ToString_AfterConstruction_ContainsComponentInfo )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Small );
        std::string output = fixture.component->toString();

        EXPECT_NE( output.find( "Softmax" ), std::string::npos );
        EXPECT_NE( output.find( "Device:" ), std::string::npos );
        EXPECT_NE( output.find( "Axis:" ), std::string::npos );
    }

    // ====================================================================
    // Forward Pass Tests - Basic Functionality
    // ====================================================================

    TEST_F( SoftmaxCpuTests, Forward_BeforeBuild_ThrowsRuntimeError )
    {
        auto fixture = SoftmaxTestFixture::Create( TestShape::Medium() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> output( Device::Cpu(), fixture.shape() );

        EXPECT_THROW(
            fixture.component->forward( input, output ),
            std::runtime_error
        );
    }

    TEST_F( SoftmaxCpuTests, Forward_WithSmallShape_ProducesValidOutput )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Small );
        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> output( Device::Cpu(), fixture.shape() );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    TEST_F( SoftmaxCpuTests, Forward_WithMediumShape_ProducesValidOutput )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Medium );
        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> output( Device::Cpu(), fixture.shape() );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    TEST_F( SoftmaxCpuTests, Forward_WithLargeShape_ProducesValidOutput )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Large );
        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> output( Device::Cpu(), fixture.shape() );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    TEST_F( SoftmaxCpuTests, Forward_MultipleIterations_ProducesConsistentResults )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Medium );
        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> output( Device::Cpu(), fixture.shape() );

        for ( int iter = 0; iter < 10; ++iter )
        {
            random( input, -5.0f, 5.0f );

            EXPECT_NO_THROW( fixture.component->forward( input, output ) )
                << "Forward pass failed at iteration " << iter;
        }
    }

    // ====================================================================
    // Forward Pass Tests - Normalization Validation
    // ====================================================================

    TEST_F( SoftmaxCpuTests, Forward_WithDefaultAxis_ProducesNormalizedOutput )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Medium );
        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> output( Device::Cpu(), fixture.shape() );

        random( input, -5.0f, 5.0f );

        fixture.component->forward( input, output );

        ValidateNormalization( output, fixture.axis );
    }

    // ====================================================================
    // Forward Pass Tests - Different Axes
    // ====================================================================

    TEST_F( SoftmaxCpuTests, Forward_WithAxis0_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        auto fixture = SoftmaxTestFixture::Create( test_shape, 0 );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> output( Device::Cpu(), fixture.shape() );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( SoftmaxCpuTests, Forward_WithAxis1_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        auto fixture = SoftmaxTestFixture::Create( test_shape, 1 );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> output( Device::Cpu(), fixture.shape() );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( SoftmaxCpuTests, Forward_WithAxis2_ProducesValidOutput )
    {
        TestShape test_shape = TestShape::Small();
        auto fixture = SoftmaxTestFixture::Create( test_shape, 2 );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> output( Device::Cpu(), fixture.shape() );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    TEST_F( SoftmaxCpuTests, Forward_WithMinimalShape_ProducesValidOutput )
    {
        TestShape minimal_shape = TestShape::Minimal();
        auto fixture = SoftmaxTestFixture::Create( minimal_shape );

        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> output( Device::Cpu(), fixture.shape() );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( SoftmaxCpuTests, Forward_WithLargeVocabulary_ProducesValidOutput )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Large );
        fixture.component->build( fixture.shape() );

        CpuTensor<TensorDataType::FP32> input( Device::Cpu(), fixture.shape() );
        CpuTensor<TensorDataType::FP32> output( Device::Cpu(), fixture.shape() );

        random( input, -5.0f, 5.0f );

        EXPECT_NO_THROW( fixture.component->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    // ====================================================================
    // Synchronization Tests
    // ====================================================================

    TEST_F( SoftmaxCpuTests, Synchronize_AfterConstruction_Succeeds )
    {
        auto& fixture = GetInferenceFixture( TestShapeSize::Small );

        EXPECT_NO_THROW( fixture.component->synchronize() );
    }
}