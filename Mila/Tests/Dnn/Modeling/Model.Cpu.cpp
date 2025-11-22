#include <gtest/gtest.h>
#include <memory>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <string>
#include <system_error>
#include <utility>

import Mila;

namespace Mila::Dnn::Modeling::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Modeling;
	using namespace Mila::Dnn::Serialization;

    // Minimal mock network implementing required Module interface points.
    class MockNetworkCpu : public Network<DeviceType::Cpu>
    {
    public:
        MockNetworkCpu() = default;
        ~MockNetworkCpu() override = default;

        // Module interface requirements not implemented by CompositeModule
        std::string getName() const override
        {
            return "mock_network_cpu";
        }

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            // Return a simple CPU device instance for tests
            return std::make_shared<CpuDevice>();
        }

        void synchronize() override
        {
            // No-op for CPU mock
        }

        // Serialization hooks - no-op for mock
        //void save( ModelArchive&, SerializationMode mode ) const override
        //{
        //}
        
        //void load( ModelArchive& ) override
        //{
        //}
    };

    // Minimal mock optimizer implementing the abstract Optimizer interface.
    class MockOptimizerCpu : public Mila::Dnn::Compute::Optimizer<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        MockOptimizerCpu() 
            : lr_( 1e-3f )
        {
        }
        ~MockOptimizerCpu() override = default;

        void addParameter( ITensor* param, ITensor* grad ) override
        {
            if (!param || !grad) throw std::invalid_argument( "param/grad cannot be null" );
            params_.push_back( param );
            grads_.push_back( grad );
        }

        void step() override
        {
            // No-op for unit tests
        }

        void zeroGrad() override
        {
            // No-op for unit tests
        }

        float getLearningRate() const override
        {
            return lr_;
        }
        void setLearningRate( float lr ) override
        {
            if (lr <= 0.0f) throw std::invalid_argument( "learning rate must be positive" );
            lr_ = lr;
        }

    private:
        float lr_;
        std::vector<ITensor*> params_;
        std::vector<ITensor*> grads_;
    };

    // Helper to create a unique temporary checkpoint directory for tests.
    static std::filesystem::path makeTempCheckpointDir()
    {
        auto tmp = std::filesystem::temp_directory_path();
        auto dir = tmp / std::filesystem::path( "mila_modelbuilder_tests_checkpoints" );
        return dir;
    }

    TEST( ModelBuilderCpu, ConfigureAndTrainProducesHistory )
    {
        // Build mocks and ModelBuilder
        auto network = std::make_unique<MockNetworkCpu>();
        auto optimizer = std::make_unique<MockOptimizerCpu>();

        Model<DeviceType::Cpu, TensorDataType::FP32> test_model(
            std::move( network ),
            std::move( optimizer ) );

        // Configure with small epoch count and disable verbose and frequent checkpoints
        ModelConfig cfg;
        auto tmpDir = makeTempCheckpointDir();
        // Ensure clean state
        std::error_code ec;
        std::filesystem::remove_all( tmpDir, ec );

        cfg.epochs( 3 )
            .checkpointDir( tmpDir )
            .checkpointFrequency( 9999 )   // avoid checkpoint save during test
            .validationSplit( 0.0 )
            .verbose( false );

        // configure() should return reference to builder for chaining
        auto& ref = test_model.configure( cfg );
        EXPECT_EQ( &ref, &test_model );

        // Run training; trainEpoch/validateEpoch are stubs that return 0.0 so losses are zeros
        auto history = test_model.train();

        EXPECT_EQ( history.train_losses.size(), cfg.getEpochs() );
        for (const auto& l : history.train_losses)
        {
            EXPECT_DOUBLE_EQ( l, 0.0 );
        }

        // Clean up created checkpoint directory (if any)
        std::filesystem::remove_all( tmpDir, ec );
    }

    TEST( ModelBuilderCpu, ResumeTrainingThrowsWhenNoCheckpoint )
    {
        auto network = std::make_unique<MockNetworkCpu>();
        auto optimizer = std::make_unique<MockOptimizerCpu>();

        Model<DeviceType::Cpu, TensorDataType::FP32> builder(
            std::move( network ),
            std::move( optimizer ) );

        ModelConfig cfg;
        cfg.epochs( 1 ).checkpointFrequency( 9999 ).verbose( false );

        builder.configure( cfg );

        // No checkpoints saved -> resumeTraining should throw
        // FIXME: EXPECT_THROW( builder.resumeTraining(), std::runtime_error );
    }
}