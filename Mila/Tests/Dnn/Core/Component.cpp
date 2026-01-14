#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

import Mila;

namespace Dnn::Components::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    class MockComponentConfig : public ComponentConfig
    {
    public:
        MockComponentConfig()
        {}

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>( getPrecisionPolicy() ) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto p = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_ )>( *p );
            }
        }

        void validate() const override
        {}

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "MockComponentConfig( precision_policy=" << static_cast<int>(getPrecisionPolicy())
                << ")";
            return oss.str();
        }
    };

    /**
     * Mock component adapted to the new Component base API.
     *
     * - Supports construction without a device (no execution context set).
     * - Supports construction from a DeviceId (owns the created execution context).
     *
     * Derived class calls the protected Component::setExecutionContext() when an
     * execution context is available.
     */
    template<DeviceType TDeviceType, TensorDataType TPrecision>
    class MockComponent : public Component<TDeviceType, TPrecision>
    {
    public:
        using ComponentBase = Component<TDeviceType, TPrecision>;

        explicit MockComponent( const MockComponentConfig& config )
            : ComponentBase( "mock_component" ), config_( config )
        {
            config_.validate();
        }

        MockComponent( const MockComponentConfig& config, std::optional<DeviceId> device_id )
            : ComponentBase( "mock_component" ), config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                // Own an execution context when constructed from DeviceId.
                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        // ====================================================================
        // Component interface implementation
        // ====================================================================

        void forward( const ITensor& input, ITensor& output )
        {
            validateTensorDevice( input, "input" );
            validateTensorDevice( output, "output" );

            if ( input.getDataType() != output.getDataType() )
            {
                throw std::runtime_error( "Input and output data types must match" );
            }

            if ( input.size() != output.size() )
            {
                throw std::runtime_error( "Input and output tensors must have the same size" );
            }
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            validateTensorDevice( input, "input" );
            validateTensorDevice( output_grad, "output_grad" );
            validateTensorDevice( input_grad, "input_grad" );

            if ( output_grad.getDataType() != input_grad.getDataType() )
            {
                throw std::runtime_error( "Gradient data types must match" );
            }

            if ( output_grad.size() != input_grad.size() )
            {
                throw std::runtime_error( "Gradient tensors must have the same size" );
            }
        }

        void synchronize() override
        {
            this->getExecutionContext()->synchronize();
        }

        size_t parameterCount() const override
        {
            return 0;
        }

        std::vector<ITensor*> getParameters() const override
        {
            return {};
        }

        std::vector<ITensor*> getGradients() const override
        {
            return {};
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            (void)archive;
            (void)mode;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "MockComponent: " << this->getName() << std::endl;
            oss << "Training: " << (this->isTraining() ? "true" : "false") << std::endl;
            return oss.str();
        }

        DeviceId getDeviceId() const override
        {
            return this->getExecutionContext()->getDeviceId();
        }

        // ====================================================================
        // Test-specific helpers
        // ====================================================================

        ComputePrecision::Policy getPrecisionPolicy() const
        {
            return config_.getPrecisionPolicy();
        }

        std::string testParametersToString() const
        {
            return parametersToString();
        }

        std::string testStateToString() const
        {
            return stateToString();
        }

    protected:

        void onBuilding( const shape_t& /*input_shape*/ ) override
        {}

    private:

        void validateTensorDevice( const ITensor& tensor, const char* name ) const
        {
            if ( tensor.getDeviceType() != TDeviceType )
            {
                throw std::invalid_argument( std::string( name ) + ": tensor device type mismatch" );
            }
        }

        std::string parametersToString() const
        {
            std::ostringstream oss;
            oss << "Parameters:\n";
            oss << "Total parameter count: " << parameterCount() << std::endl;
            return oss.str();
        }

        std::string stateToString() const
        {
            return std::string();
        }

        MockComponentConfig config_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };
    };

    /**
     * Test data factory updated to work with the new MockComponent constructors.
     * Factories continue to provide both "create from DeviceId" and "create with
     * an owned execution context" flows.
     */
    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    struct ComponentTestData
    {
        std::shared_ptr<MockComponent<TDeviceType, TPrecision>> component;
        bool is_training;
        std::unique_ptr<IExecutionContext> exec_context;

        static ComponentTestData Create(
            int device_id,
            bool is_training = false )
        {
            ComponentTestData data;
            data.is_training = is_training;

            MockComponentConfig config;

            if constexpr ( TDeviceType == DeviceType::Cuda )
            {
                data.exec_context = createExecutionContext( Device::Cuda( device_id ) );
            }
            else
            {
                data.exec_context = createExecutionContext( Device::Cpu() );
            }

            // Construct component from the DeviceId (component will own its exec context)
            data.component = std::make_shared<MockComponent<TDeviceType, TPrecision>>(
                config,
                std::optional<DeviceId>( data.exec_context->getDeviceId() )
            );

            if ( data.is_training )
            {
                // Build before enabling training to satisfy Component lifecycle contract.
                shape_t build_shape = { 2, 3 };
                data.component->build( build_shape );

                data.component->setTraining( true );
            }

            return data;
        }

        static ComponentTestData CreateWithContext(
            std::unique_ptr<IExecutionContext> exec_context,
            bool is_training = false )
        {
            ComponentTestData data;
            data.is_training = is_training;

            MockComponentConfig config;

            data.exec_context = std::move( exec_context );

            // Construct component from the DeviceId (component will own its exec context)
            data.component = std::make_shared<MockComponent<TDeviceType, TPrecision>>(
                config,
                std::optional<DeviceId>( data.exec_context->getDeviceId() )
            );

            if ( data.is_training )
            {
                // Build before enabling training to satisfy Component lifecycle contract.
                shape_t build_shape = { 2, 3 };
                data.component->build( build_shape );

                data.component->setTraining( true );
            }

            return data;
        }
    };

    class ComponentTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {}

        void TearDown() override
        {
            cpu_data_.component.reset();
            cuda_data_.component.reset();
            training_cpu_data_.component.reset();
            training_cuda_data_.component.reset();
            context_cpu_data_.component.reset();
            context_cuda_data_.component.reset();
        }

        ComponentTestData<DeviceType::Cpu>& CpuData()
        {
            if ( !cpu_data_.component )
            {
                cpu_data_ = ComponentTestData<DeviceType::Cpu>::Create( 0 );
            }
            return cpu_data_;
        }

        ComponentTestData<DeviceType::Cuda>& CudaData()
        {
            if ( !cuda_data_.component )
            {
                cuda_data_ = ComponentTestData<DeviceType::Cuda>::Create( 0 );
            }
            return cuda_data_;
        }

        ComponentTestData<DeviceType::Cpu>& TrainingCpuData()
        {
            if ( !training_cpu_data_.component )
            {
                training_cpu_data_ = ComponentTestData<DeviceType::Cpu>::Create( 0, true );
            }
            return training_cpu_data_;
        }

        ComponentTestData<DeviceType::Cuda>& TrainingCudaData()
        {
            if ( !training_cuda_data_.component )
            {
                training_cuda_data_ = ComponentTestData<DeviceType::Cuda>::Create( 0, true );
            }
            return training_cuda_data_;
        }

        ComponentTestData<DeviceType::Cpu>& ContextCpuData()
        {
            if ( !context_cpu_data_.component )
            {
                auto exec_context = createExecutionContext( Device::Cpu() );
                context_cpu_data_ = ComponentTestData<DeviceType::Cpu>::CreateWithContext( std::move( exec_context ) );
            }
            return context_cpu_data_;
        }

        ComponentTestData<DeviceType::Cuda>& ContextCudaData()
        {
            if ( !context_cuda_data_.component )
            {
                auto exec_context = createExecutionContext( Device::Cuda( 0 ) );
                context_cuda_data_ = ComponentTestData<DeviceType::Cuda>::CreateWithContext( std::move( exec_context ) );
            }
            return context_cuda_data_;
        }

        ComponentTestData<DeviceType::Cpu> cpu_data_;
        ComponentTestData<DeviceType::Cuda> cuda_data_;
        ComponentTestData<DeviceType::Cpu> training_cpu_data_;
        ComponentTestData<DeviceType::Cuda> training_cuda_data_;
        ComponentTestData<DeviceType::Cpu> context_cpu_data_;
        ComponentTestData<DeviceType::Cuda> context_cuda_data_;
    };

    // ====================================================================
    // Test Helper Functions
    // ====================================================================

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestGetName( const ComponentTestData<TDeviceType, TPrecision>& data, const std::string& expected_name )
    {
        EXPECT_EQ( data.component->getName(), expected_name );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestParameterCount( const ComponentTestData<TDeviceType, TPrecision>& data, size_t expected_count )
    {
        EXPECT_EQ( data.component->parameterCount(), expected_count );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestPrint( const ComponentTestData<TDeviceType, TPrecision>& data, const std::string& expected_substring )
    {
        std::string output = data.component->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestTrainingMode( const ComponentTestData<TDeviceType, TPrecision>& data, bool expected_mode )
    {
        EXPECT_EQ( data.component->isTraining(), expected_mode );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestDeviceType( const ComponentTestData<TDeviceType, TPrecision>& data )
    {
        auto device = data.component->getDeviceId();

        EXPECT_EQ( device.type, TDeviceType );
        EXPECT_EQ( data.component->getDeviceType(), TDeviceType );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestPrecision( const ComponentTestData<TDeviceType, TPrecision>& data )
    {
        EXPECT_EQ( data.component->getPrecisionPolicy(), ComputePrecision::Policy::Auto );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestHelperMethods( const ComponentTestData<TDeviceType, TPrecision>& data )
    {
        std::string params_str = data.component->testParametersToString();
        std::string state_str = data.component->testStateToString();

        EXPECT_EQ( params_str, "Parameters:\nTotal parameter count: 0\n" );
        EXPECT_EQ( state_str, "" );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestForwardBackward( const ComponentTestData<TDeviceType, TPrecision>& data )
    {
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda,
            CudaDeviceMemoryResource,
            CpuMemoryResource>;

        std::vector<int64_t> shape = { 2, 3 };

        auto exec_ctx = data.exec_context.get();
        auto device = exec_ctx->getDeviceId();

        Tensor<TensorDataType::FP32, MR> input( device, shape );
        Tensor<TensorDataType::FP32, MR> output( device, shape );
        Tensor<TensorDataType::FP32, MR> input_grad( device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( device, shape );

        if constexpr ( TDeviceType == DeviceType::Cpu )
        {
            auto* input_data = input.data();
            auto* grad_data = output_grad.data();

            for ( size_t i = 0; i < input.size(); ++i )
            {
                input_data[ i ] = static_cast<float>( i + 1.0f );
                grad_data[ i ] = 0.5f;
            }
        }
        else
        {
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( Device::Cpu(), shape );
            Tensor<TensorDataType::FP32, CpuMemoryResource> host_grad( Device::Cpu(), shape );

            auto* host_input_data = host_input.data();
            auto* host_grad_data = host_grad.data();

            for ( size_t i = 0; i < host_input.size(); ++i )
            {
                host_input_data[ i ] = static_cast<float>( i + 1.0f );
                host_grad_data[ i ] = 0.5f;
            }

            exec_ctx->synchronize();
        }

        EXPECT_NO_THROW( data.component->forward( input, output ) );
        EXPECT_NO_THROW( data.component->backward( input, output_grad, input_grad ) );

        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( input_grad.size(), output_grad.size() );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestInvalidDeviceId()
    {
        MockComponentConfig config;

        if constexpr ( TDeviceType == DeviceType::Cuda )
        {
            EXPECT_THROW(
                (createExecutionContext( Device::Cuda( -1 ) )),
                std::invalid_argument
            );
        }
        else
        {
            // Constructing with a valid Cpu DeviceId should not throw.
            DeviceId cpu_id = Device::Cpu();
            EXPECT_NO_THROW(
                (std::make_shared<MockComponent<TDeviceType, TPrecision>>( config, std::optional<DeviceId>( cpu_id ) ))
            );
        }
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestNullExecutionContext()
    {
        MockComponentConfig config;

        // Construct without device id is allowed; calling getDeviceId() should throw.
        auto comp = std::make_shared<MockComponent<TDeviceType, TPrecision>>( config );

        EXPECT_THROW(
            comp->getDeviceId(),
            std::runtime_error
        );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision = TensorDataType::FP32>
    void TestDeviceTypeMismatch()
    {
        MockComponentConfig config;

        if constexpr ( TDeviceType == DeviceType::Cuda )
        {
            auto cpu_ctx = createExecutionContext( Device::Cpu() );
            DeviceId cpu_id = cpu_ctx->getDeviceId();

            EXPECT_THROW(
                (std::make_shared<MockComponent<TDeviceType, TPrecision>>( config, std::optional<DeviceId>( cpu_id ) )),
                std::invalid_argument
            );
        }
        else
        {
            auto cuda_ctx = createExecutionContext( Device::Cuda( 0 ) );
            DeviceId cuda_id = cuda_ctx->getDeviceId();

            EXPECT_THROW(
                (std::make_shared<MockComponent<TDeviceType, TPrecision>>( config, std::optional<DeviceId>( cuda_id ) )),
                std::invalid_argument
            );
        }
    }

    // ====================================================================
    // CPU Tests
    // ====================================================================

    TEST_F( ComponentTests, Cpu_GetName )
    {
        TestGetName( CpuData(), "mock_component" );
    }

    TEST_F( ComponentTests, Cpu_ParameterCount )
    {
        TestParameterCount( CpuData(), 0 );
    }

    TEST_F( ComponentTests, Cpu_Print )
    {
        TestPrint( CpuData(), "MockComponent: mock_component" );
    }

    TEST_F( ComponentTests, Cpu_TrainingMode )
    {
        TestTrainingMode( CpuData(), false );
    }

    TEST_F( ComponentTests, Cpu_DeviceType )
    {
        TestDeviceType( CpuData() );
    }

    TEST_F( ComponentTests, Cpu_Precision )
    {
        TestPrecision( CpuData() );
    }

    TEST_F( ComponentTests, Cpu_HelperMethods )
    {
        TestHelperMethods( CpuData() );
    }

    TEST_F( ComponentTests, Cpu_ForwardBackward )
    {
        TestForwardBackward( CpuData() );
    }

    TEST_F( ComponentTests, Cpu_Training_TrainingMode )
    {
        TestTrainingMode( TrainingCpuData(), true );
    }

    TEST_F( ComponentTests, Context_Cpu_DeviceType )
    {
        TestDeviceType( ContextCpuData() );
    }

    // ====================================================================
    // CUDA Tests
    // ====================================================================

    TEST_F( ComponentTests, Cuda_GetName )
    {
        TestGetName( CudaData(), "mock_component" );
    }

    TEST_F( ComponentTests, Cuda_ParameterCount )
    {
        TestParameterCount( CudaData(), 0 );
    }

    TEST_F( ComponentTests, Cuda_Print )
    {
        TestPrint( CudaData(), "MockComponent: mock_component" );
    }

    TEST_F( ComponentTests, Cuda_TrainingMode )
    {
        TestTrainingMode( CudaData(), false );
    }

    TEST_F( ComponentTests, Cuda_DeviceType )
    {
        TestDeviceType( CudaData() );
    }

    TEST_F( ComponentTests, Cuda_Precision )
    {
        TestPrecision( CudaData() );
    }

    TEST_F( ComponentTests, Cuda_ForwardBackward )
    {
        TestForwardBackward( CudaData() );
    }

    TEST_F( ComponentTests, Cuda_Training_TrainingMode )
    {
        TestTrainingMode( TrainingCudaData(), true );
    }

    TEST_F( ComponentTests, Context_Cuda_DeviceType )
    {
        TestDeviceType( ContextCudaData() );
    }

    // ====================================================================
    // Configuration Tests
    // ====================================================================

    TEST_F( ComponentTests, ComponentConfig_DefaultValues )
    {
        MockComponentConfig config;
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Auto );
    }

    TEST_F( ComponentTests, ComponentConfig_CustomValues )
    {
        MockComponentConfig config;
        config.withPrecisionPolicy( ComputePrecision::Policy::Performance );

        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
    }

    // ====================================================================
    // Error Handling Tests
    // ====================================================================

    TEST_F( ComponentTests, InvalidDeviceId_Cpu )
    {
        TestInvalidDeviceId<DeviceType::Cpu>();
    }

    TEST_F( ComponentTests, InvalidDeviceId_Cuda )
    {
        TestInvalidDeviceId<DeviceType::Cuda>();
    }

    TEST_F( ComponentTests, NullExecutionContext_Cpu )
    {
        TestNullExecutionContext<DeviceType::Cpu>();
    }

    TEST_F( ComponentTests, NullExecutionContext_Cuda )
    {
        TestNullExecutionContext<DeviceType::Cuda>();
    }

    TEST_F( ComponentTests, DeviceTypeMismatch_Cpu )
    {
        TestDeviceTypeMismatch<DeviceType::Cpu>();
    }

    TEST_F( ComponentTests, DeviceTypeMismatch_Cuda )
    {
        TestDeviceTypeMismatch<DeviceType::Cuda>();
    }
}