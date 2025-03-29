#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

import Mila;

namespace Modules::Tests
{
    using namespace Mila::Dnn;

    class GeluTests : public ::testing::Test {
    protected:
        void SetUp() override {
			batch_size_ = 64;
			cpu_batch_size_= 4;
			sequence_length_ = 1024;
			channels_ = 768;
			cpu_io_shape_ = { cpu_batch_size_, sequence_length_, 4 * channels_ };
			//cuda_io_shape_ = { batch_size_, sequence_length_, 4 * channels_ };

            cpu_gelu = std::make_shared<Gelu<float, float, Compute::DeviceType::Cpu>>(
                "cpu_gelu" );

            /*cuda_gelu = std::make_unique<Gelu<float, float, Compute::DeviceType::Cuda>>(
                "cuda_gelu" );*/
        }

        std::shared_ptr<Gelu<float, float, Compute::DeviceType::Cpu>> cpu_gelu;
        //std::unique_ptr<Gelu<float, float, Compute::DeviceType::Cuda>> cuda_gelu;
        
        size_t batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t sequence_length_{ 0 };
        size_t channels_{ 0 };
		std::vector<size_t> cpu_io_shape_;
		//std::vector<size_t> cuda_io_shape_;
    };

    TEST_F( GeluTests, Cpu_TestName ) {
        EXPECT_EQ( cpu_gelu->getName(), "cpu_gelu" );
    }

    TEST_F( GeluTests, Cpu_parameterCount ) {
        auto num_parameters = 0;
        EXPECT_EQ( cpu_gelu->parameterCount(), num_parameters ) ;
    }

    TEST_F( GeluTests, Cpu_TestForward ) {
        Tensor<float, Compute::HostMemoryResource> input( cpu_io_shape_ );
        Tensor<float, Compute::HostMemoryResource> output( cpu_io_shape_ );
        cpu_gelu->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );
    }

    /*TEST_F( GeluTests, Cuda_TestForward ) {
        Tensor<float, Compute::DeviceMemoryResource> input( cuda_io_shape_ );
        Tensor<float, Compute::DeviceMemoryResource> output( cuda_io_shape_ );
        cuda_gelu->forward( input, output );
        EXPECT_EQ( output.size(), input.size() );
    }*/

    TEST_F( GeluTests, Cpu_TestPrint ) {
		std::string output = cpu_gelu->toString();
        EXPECT_NE( output.find( "Gelu: cpu_gelu" ), std::string::npos );
    }
}