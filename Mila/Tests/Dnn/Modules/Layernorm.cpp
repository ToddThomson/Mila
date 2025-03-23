#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    class LayerNormTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_size_ = 128;
            cpu_batch_size_ = 4;
            sequence_length_ = 1024;
            channels_ = 768;
			cpu_io_shape_ = { cpu_batch_size_, sequence_length_, channels_ };
            cuda_input_shape_ = { batch_size_, sequence_length_, channels_ };
            has_bias_ = true;

            cpu_layernorm = std::make_unique<LayerNorm<float, float, Compute::DeviceType::Cpu>>(
                "cpu_ln", cpu_io_shape_ );

            /*cuda_linear = std::make_unique<MilaDnn::Modules::Linear<float, MilaDnn::Compute::DeviceMemoryResource>>(
                "cuda_ln", input_shape_, output_channels_ );*/
        }

        std::unique_ptr<LayerNorm<float, float, Compute::DeviceType::Cpu>> cpu_layernorm;
        std::unique_ptr<LayerNorm<float, float, Compute::DeviceType::Cuda>> cuda_layernorm;

        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
        std::vector<size_t> cpu_io_shape_;
        std::vector<size_t> cuda_input_shape_;
        bool has_bias_{ true };
    };

    TEST_F( LayerNormTests, Cpu_TestName ) {
        EXPECT_EQ( cpu_layernorm->getName(), "cpu_ln" );
    }

    TEST_F( LayerNormTests, Cpu_parameterCount ) {
        auto num_parameters = /* weights */ channels_;
        if ( has_bias_ ) {
            num_parameters += channels_;
        }

        EXPECT_EQ( cpu_layernorm->parameterCount(), num_parameters );
    }

    TEST_F( LayerNormTests, Cpu_TestWeightInitialization ) {
        auto weight = cpu_layernorm->getWeight();
        EXPECT_EQ( weight->size(), channels_ );
    }

    TEST_F( LayerNormTests, Cpu_TestBiasInitialization ) {
        auto bias = cpu_layernorm->getBias();
        EXPECT_EQ( bias->size(), channels_ );
    }

    TEST_F( LayerNormTests, Cpu_TestForward ) {
		std::vector<size_t> io_shape = { 1, 2, 3 };
        Tensor<float, Compute::CpuMemoryResource> input( io_shape );
        Tensor<float, Compute::CpuMemoryResource> output( io_shape );

        input.data()[ 0 ] = 1.0f;
        input.data()[ 1 ] = 2.0f;
        input.data()[ 2 ] = 3.0f;
        input.data()[ 3 ] = 4.0f;
        input.data()[ 4 ] = 5.0f;
        input.data()[ 5 ] = 6.0f;

        auto ln = std::make_unique<LayerNorm<float, float, Compute::DeviceType::Cpu>>(
            "ln", io_shape );
        
        ln->forward( input, output );

        // Verify the output tensor  
        EXPECT_NEAR( output.data()[ 0 ], -1.22474f, 1e-5 );
        EXPECT_NEAR( output.data()[ 1 ], 0.0f, 1e-5 );
        EXPECT_NEAR( output.data()[ 2 ], 1.22474f, 1e-5 );
        EXPECT_NEAR( output.data()[ 3 ], -1.22474f, 1e-5 );
        EXPECT_NEAR( output.data()[ 4 ], 0.0f, 1e-5 );
        EXPECT_NEAR( output.data()[ 5 ], 1.22474f, 1e-5 );

        EXPECT_EQ( output.size(), input.size() );
    }

    TEST( CpuLayerNormOpTest, ForwardSimple3DInput ) {
        
    }

    /*TEST_F( LayerNormTests, Cuda_TestForward ) {
        MilaDnn::Tensor<float, MilaDnn::Compute::DeviceMemoryResource> input( { batch_size_, sequence_length_, channels_ } );
        auto output = cuda_linear->forward( std::make_shared<MilaDnn::Tensor<float, MilaDnn::Compute::DeviceMemoryResource>>( input ) );
        EXPECT_EQ( output->size(), batch_size_ * sequence_length_ * output_channels_ );
    }*/

    TEST_F( LayerNormTests, Cpu_TestPrint ) {
        std::string output = cpu_layernorm->toString();
        EXPECT_NE( output.find( "LayerNorm: cpu_ln" ), std::string::npos );
    }
}