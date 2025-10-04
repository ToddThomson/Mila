#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cuda_fp16.h>
#include <type_traits>

import Mila;

namespace Modules::Blocks::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    template<DeviceType TDevice, typename TDataType>
    using MemoryResourceType = std::conditional_t<TDevice == DeviceType::Cuda,
        CudaDeviceMemoryResource,
        CpuMemoryResource>;

    template<DeviceType TDevice, typename TDataType = float>
    struct MLPTestData {
        MLPConfig config;
        std::shared_ptr<MLP<TDevice, TDataType>> mlp_module;
        std::vector<size_t> input_shape;
        size_t hidden_size;

        MLPTestData() : config( { 1 }, 1 ) {}

        static MLPTestData Create(
            const std::string& name,
            const std::vector<size_t>& input_shape,
            size_t hidden_size,
            bool has_bias = true,
            bool is_training = false,
            ActivationType activation = ActivationType::Gelu,
            bool use_layer_norm = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            MLPTestData data;
            data.input_shape = input_shape;
            data.hidden_size = hidden_size;

            data.config = MLPConfig( input_shape, hidden_size )
                .withBias( has_bias )
                .withActivation( activation )
                .withLayerNorm( use_layer_norm )
                .withName( name )
                .withPrecisionPolicy( precision )
                .withTraining( is_training );

            std::string device_str = TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU";
            data.mlp_module = std::make_shared<MLP<TDevice, TDataType>>( device_str, data.config );

            return data;
        }

        static MLPTestData CreateWithContext(
            const std::string& name,
            const std::vector<size_t>& input_shape,
            size_t hidden_size,
            std::shared_ptr<DeviceContext> context,
            bool has_bias = true,
            bool is_training = false,
            ActivationType activation = ActivationType::Gelu,
            bool use_layer_norm = false,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto )
        {
            MLPTestData data;
            data.input_shape = input_shape;
            data.hidden_size = hidden_size;

            data.config = MLPConfig( input_shape, hidden_size )
                .withBias( has_bias )
                .withActivation( activation )
                .withLayerNorm( use_layer_norm )
                .withName( name )
                .withPrecisionPolicy( precision )
                .withTraining( is_training );

            data.mlp_module = std::make_shared<MLP<TDevice, TDataType>>( context, data.config );

            return data;
        }
    };

    class MLPTests : public ::testing::Test {
    protected:
        void SetUp() override {
            cuda_batch_size_ = 16;
            cuda_sequence_length_ = 64;
            cpu_batch_size_ = 2;
            cpu_sequence_length_ = 16;
            input_features_ = 768;
            hidden_size_ = 3072;
        }

        void TearDown() override {
            cpu_float_data_.mlp_module.reset();
            context_cpu_float_data_.mlp_module.reset();
            training_cpu_float_data_.mlp_module.reset();
            no_bias_cpu_float_data_.mlp_module.reset();
            layer_norm_cpu_float_data_.mlp_module.reset();

            cuda_float_data_.mlp_module.reset();
            training_cuda_float_data_.mlp_module.reset();
            no_bias_cuda_float_data_.mlp_module.reset();
            layer_norm_cuda_float_data_.mlp_module.reset();

            cuda_half_data_.mlp_module.reset();
            training_cuda_half_data_.mlp_module.reset();

            perf_precision_cuda_float_data_.mlp_module.reset();
            accuracy_precision_cuda_float_data_.mlp_module.reset();
            native_precision_cuda_float_data_.mlp_module.reset();
        }

        MLPTestData<DeviceType::Cpu, float>& CpuFloatData() {
            if ( !cpu_float_data_.mlp_module ) {
                cpu_float_data_ = MLPTestData<DeviceType::Cpu, float>::Create(
                    "cpu_mlp_float", { cpu_batch_size_, cpu_sequence_length_, input_features_ }, hidden_size_ );
            }
            return cpu_float_data_;
        }

        MLPTestData<DeviceType::Cuda, float>& CudaFloatData() {
            if ( !cuda_float_data_.mlp_module ) {
                cuda_float_data_ = MLPTestData<DeviceType::Cuda, float>::Create(
                    "cuda_mlp_float", { cuda_batch_size_, cuda_sequence_length_, input_features_ }, hidden_size_ );
            }
            return cuda_float_data_;
        }

        MLPTestData<DeviceType::Cpu, float>& TrainingCpuFloatData() {
            if ( !training_cpu_float_data_.mlp_module ) {
                training_cpu_float_data_ = MLPTestData<DeviceType::Cpu, float>::Create(
                    "cpu_mlp_float_training", { cpu_batch_size_, cpu_sequence_length_, input_features_ },
                    hidden_size_, true, true );
            }
            return training_cpu_float_data_;
        }

        MLPTestData<DeviceType::Cuda, float>& TrainingCudaFloatData() {
            if ( !training_cuda_float_data_.mlp_module ) {
                training_cuda_float_data_ = MLPTestData<DeviceType::Cuda, float>::Create(
                    "cuda_mlp_float_training", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                    hidden_size_, true, true );
            }
            return training_cuda_float_data_;
        }

        MLPTestData<DeviceType::Cpu, float>& NoBiasCpuFloatData() {
            if ( !no_bias_cpu_float_data_.mlp_module ) {
                no_bias_cpu_float_data_ = MLPTestData<DeviceType::Cpu, float>::Create(
                    "cpu_mlp_float_nobias", { cpu_batch_size_, cpu_sequence_length_, input_features_ },
                    hidden_size_, false );
            }
            return no_bias_cpu_float_data_;
        }

        MLPTestData<DeviceType::Cuda, float>& NoBiasCudaFloatData() {
            if ( !no_bias_cuda_float_data_.mlp_module ) {
                no_bias_cuda_float_data_ = MLPTestData<DeviceType::Cuda, float>::Create(
                    "cuda_mlp_float_nobias", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                    hidden_size_, false );
            }
            return no_bias_cuda_float_data_;
        }

        MLPTestData<DeviceType::Cpu, float>& LayerNormCpuFloatData() {
            if ( !layer_norm_cpu_float_data_.mlp_module ) {
                layer_norm_cpu_float_data_ = MLPTestData<DeviceType::Cpu, float>::Create(
                    "cpu_mlp_float_layernorm", { cpu_batch_size_, cpu_sequence_length_, input_features_ },
                    hidden_size_, true, false, ActivationType::Gelu, true );
            }
            return layer_norm_cpu_float_data_;
        }

        MLPTestData<DeviceType::Cuda, float>& LayerNormCudaFloatData() {
            if ( !layer_norm_cuda_float_data_.mlp_module ) {
                layer_norm_cuda_float_data_ = MLPTestData<DeviceType::Cuda, float>::Create(
                    "cuda_mlp_float_layernorm", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                    hidden_size_, true, false, ActivationType::Gelu, true );
            }
            return layer_norm_cuda_float_data_;
        }

        MLPTestData<DeviceType::Cpu, float>& ContextCpuFloatData() {
            if ( !context_cpu_float_data_.mlp_module ) {
                auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
                context_cpu_float_data_ = MLPTestData<DeviceType::Cpu, float>::CreateWithContext(
                    "cpu_context_mlp_float", { cpu_batch_size_, cpu_sequence_length_, input_features_ },
                    hidden_size_, cpu_context );
            }
            return context_cpu_float_data_;
        }

        MLPTestData<DeviceType::Cuda, half>& CudaHalfData() {
            if ( !cuda_half_data_.mlp_module ) {
                cuda_half_data_ = MLPTestData<DeviceType::Cuda, half>::Create(
                    "cuda_mlp_half", { cuda_batch_size_, cuda_sequence_length_, input_features_ }, hidden_size_ );
            }
            return cuda_half_data_;
        }

        MLPTestData<DeviceType::Cuda, half>& TrainingCudaHalfData() {
            if ( !training_cuda_half_data_.mlp_module ) {
                training_cuda_half_data_ = MLPTestData<DeviceType::Cuda, half>::Create(
                    "cuda_mlp_half_training", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                    hidden_size_, true, true );
            }
            return training_cuda_half_data_;
        }

        MLPTestData<DeviceType::Cuda, float>& PerfPrecisionCudaFloatData() {
            if ( !perf_precision_cuda_float_data_.mlp_module ) {
                perf_precision_cuda_float_data_ = MLPTestData<DeviceType::Cuda, float>::Create(
                    "cuda_mlp_perf_precision", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                    hidden_size_, true, false, ActivationType::Gelu, false,
                    ComputePrecision::Policy::Performance );
            }
            return perf_precision_cuda_float_data_;
        }

        MLPTestData<DeviceType::Cuda, float>& AccuracyPrecisionCudaFloatData() {
            if ( !accuracy_precision_cuda_float_data_.mlp_module ) {
                accuracy_precision_cuda_float_data_ = MLPTestData<DeviceType::Cuda, float>::Create(
                    "cuda_mlp_accuracy_precision", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                    hidden_size_, true, false, ActivationType::Gelu, false,
                    ComputePrecision::Policy::Accuracy );
            }
            return accuracy_precision_cuda_float_data_;
        }

        MLPTestData<DeviceType::Cuda, float>& NativePrecisionCudaFloatData() {
            if ( !native_precision_cuda_float_data_.mlp_module ) {
                native_precision_cuda_float_data_ = MLPTestData<DeviceType::Cuda, float>::Create(
                    "cuda_mlp_native_precision", { cuda_batch_size_, cuda_sequence_length_, input_features_ },
                    hidden_size_, true, false, ActivationType::Gelu, false,
                    ComputePrecision::Policy::Native );
            }
            return native_precision_cuda_float_data_;
        }

        size_t cuda_batch_size_{ 0 };
        size_t cpu_batch_size_{ 0 };
        size_t cuda_sequence_length_{ 0 };
        size_t cpu_sequence_length_{ 0 };
        size_t input_features_{ 0 };
        size_t hidden_size_{ 0 };

        MLPTestData<DeviceType::Cpu, float> cpu_float_data_;
        MLPTestData<DeviceType::Cpu, float> context_cpu_float_data_;
        MLPTestData<DeviceType::Cpu, float> training_cpu_float_data_;
        MLPTestData<DeviceType::Cpu, float> no_bias_cpu_float_data_;
        MLPTestData<DeviceType::Cpu, float> layer_norm_cpu_float_data_;

        MLPTestData<DeviceType::Cuda, float> cuda_float_data_;
        MLPTestData<DeviceType::Cuda, float> training_cuda_float_data_;
        MLPTestData<DeviceType::Cuda, float> no_bias_cuda_float_data_;
        MLPTestData<DeviceType::Cuda, float> layer_norm_cuda_float_data_;

        MLPTestData<DeviceType::Cuda, half> cuda_half_data_;
        MLPTestData<DeviceType::Cuda, half> training_cuda_half_data_;

        MLPTestData<DeviceType::Cuda, float> perf_precision_cuda_float_data_;
        MLPTestData<DeviceType::Cuda, float> accuracy_precision_cuda_float_data_;
        MLPTestData<DeviceType::Cuda, float> native_precision_cuda_float_data_;
    };

    template<DeviceType TDevice, typename TDataType = float>
    void TestGetName( const MLPTestData<TDevice, TDataType>& data, const std::string& expected_name ) {
        EXPECT_EQ( data.mlp_module->getName(), expected_name );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestParameterCount( const MLPTestData<TDevice, TDataType>& data ) {
        size_t input_features = data.config.getInputFeatures();
        size_t hidden_size = data.config.getHiddenSize();
        bool has_bias = data.config.hasBias();

        size_t expected_fc1_params = input_features * hidden_size;
        size_t expected_fc2_params = hidden_size * input_features;

        if ( has_bias ) {
            expected_fc1_params += hidden_size;
            expected_fc2_params += input_features;
        }

        size_t expected_total_params = expected_fc1_params + expected_fc2_params;

        EXPECT_EQ( data.mlp_module->parameterCount(), expected_total_params );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestForward( const MLPTestData<TDevice, TDataType>& data ) {
        using MR = MemoryResourceType<TDevice, TDataType>;

        Tensor<TDataType, MR> input( data.input_shape );
        Tensor<TDataType, MR> output( data.input_shape );

        random<TDataType, MR>( input, static_cast<TDataType>(-1.0), static_cast<TDataType>(1.0) );

        EXPECT_NO_THROW( data.mlp_module->forward( input, output ) );

        EXPECT_EQ( output.size(), input.size() );
        EXPECT_EQ( output.shape(), input.shape() );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestToString( const MLPTestData<TDevice, TDataType>& data, const std::string& expected_substring ) {
        std::string output = data.mlp_module->toString();
        EXPECT_NE( output.find( expected_substring ), std::string::npos );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestTrainingMode( const MLPTestData<TDevice, TDataType>& data, bool expected_mode ) {
        EXPECT_EQ( data.mlp_module->isTraining(), expected_mode );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestDeviceType( const MLPTestData<TDevice, TDataType>& data ) {
        auto device_context = data.mlp_module->getDeviceContext();
        EXPECT_NE( device_context, nullptr );
        auto device = device_context->getDevice();
        EXPECT_NE( device, nullptr );
        EXPECT_EQ( device->getDeviceType(), TDevice );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestSubModules( const MLPTestData<TDevice, TDataType>& data ) {
        auto modules = data.mlp_module->getNamedModules();

        EXPECT_GE( modules.size(), 3 );
        EXPECT_NE( modules.find( "fc1" ), modules.end() );
        EXPECT_NE( modules.find( "activation" ), modules.end() );
        EXPECT_NE( modules.find( "fc2" ), modules.end() );

        if ( data.config.useLayerNorm() ) {
            EXPECT_NE( modules.find( "norm1" ), modules.end() );
        }
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestSaveLoad( const MLPTestData<TDevice, TDataType>& data ) {
        ModelArchive archive;
        EXPECT_NO_THROW( data.mlp_module->save( archive ) );
        EXPECT_NO_THROW( data.mlp_module->load( archive ) );
    }

    template<DeviceType TDevice, typename TDataType = float>
    void TestEdgeCases() {
        using MR = MemoryResourceType<TDevice, TDataType>;

        try {
            std::vector<size_t> minimal_shape = { 1, 1, 8 };
            size_t minimal_hidden_size = 16;

            auto minimal_config = MLPConfig( minimal_shape, minimal_hidden_size ).withName( "minimal_mlp" );
            std::string device_str = TDevice == DeviceType::Cuda ? "CUDA:0" : "CPU";
            auto minimal_module = std::make_shared<MLP<TDevice, TDataType>>( device_str, minimal_config );

            Tensor<TDataType, MR> minimal_input( minimal_shape );
            Tensor<TDataType, MR> minimal_output( minimal_shape );

            EXPECT_NO_THROW( minimal_module->forward( minimal_input, minimal_output ) );
            EXPECT_EQ( minimal_output.size(), 8 );

            std::vector<size_t> medium_shape;
            if constexpr ( TDevice == DeviceType::Cuda ) {
                medium_shape = { 2, 2, 1024 };
            }
            else {
                medium_shape = { 1, 2, 512 };
            }

            size_t medium_hidden_size = 2048;
            auto medium_config = MLPConfig( medium_shape, medium_hidden_size ).withName( "medium_mlp" );
            auto medium_module = std::make_shared<MLP<TDevice, TDataType>>( device_str, medium_config );

            Tensor<TDataType, MR> medium_input( medium_shape );
            Tensor<TDataType, MR> medium_output( medium_shape );

            EXPECT_NO_THROW( medium_module->forward( medium_input, medium_output ) );
        }
        catch ( const std::exception& e ) {
            std::cerr << "Exception during edge case test: " << e.what() << std::endl;
            throw;
        }
    }

    template<typename TDataType = float>
    void TestCpuCudaEquivalence(
        const MLPTestData<DeviceType::Cpu, TDataType>& cpu_data,
        const MLPTestData<DeviceType::Cuda, TDataType>& cuda_data ) {

        try {
            DeviceContext context( "CUDA:0" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping CPU-CUDA equivalence test";
            return;
        }

        std::vector<size_t> test_shape = { 1, 2, cpu_data.config.getInputFeatures() };
        size_t test_hidden_size = 1024;

        auto cpu_config = MLPConfig( test_shape, test_hidden_size ).withName( "test_cpu_mlp" );
        auto cuda_config = MLPConfig( test_shape, test_hidden_size ).withName( "test_cuda_mlp" );

        auto cpu_mlp = std::make_shared<MLP<DeviceType::Cpu, TDataType>>( "CPU", cpu_config );
        auto cuda_mlp = std::make_shared<MLP<DeviceType::Cuda, TDataType>>( "CUDA:0", cuda_config );

        Tensor<TDataType, CpuMemoryResource> host_input( test_shape );

        for ( size_t i = 0; i < host_input.size(); ++i ) {
            host_input.data()[ i ] = static_cast<TDataType>( -2.0 + 4.0 * (static_cast<float>( i ) / host_input.size()) );
        }

        Tensor<TDataType, CpuMemoryResource> cpu_output( test_shape );
        cpu_mlp->forward( host_input, cpu_output );

        Tensor<TDataType, CudaDeviceMemoryResource> device_input = host_input.toDevice<CudaDeviceMemoryResource>();

        Tensor<TDataType, CudaDeviceMemoryResource> cuda_output( test_shape );
        cuda_mlp->forward( device_input, cuda_output );

        cuda_mlp->getDeviceContext()->synchronize();

        Tensor<TDataType, CpuMemoryResource> cuda_output_host = cuda_output.toHost<CpuMemoryResource>();

        const float epsilon = 1e-3f;
        bool all_equal = true;

        for ( size_t i = 0; i < cpu_output.size(); ++i ) {
            float diff = std::abs( static_cast<float>( cpu_output.data()[ i ] ) - static_cast<float>( cuda_output_host.data()[ i ] ) );
            if ( diff > epsilon ) {
                std::cout << "Difference at index " << i << ": CPU=" << cpu_output.data()[ i ]
                    << ", CUDA=" << cuda_output_host.data()[ i ] << ", diff=" << diff << std::endl;
                all_equal = false;
            }
        }

        EXPECT_TRUE( all_equal ) << "CPU and CUDA implementations produced different results";
    }

    TEST_F( MLPTests, Cpu_Float_TestName ) {
        TestGetName<DeviceType::Cpu, float>( CpuFloatData(), "cpu_mlp_float" );
    }

    TEST_F( MLPTests, Cpu_Float_ParameterCount ) {
        TestParameterCount<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_TestForward ) {
        TestForward<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_TestToString ) {
        TestToString<DeviceType::Cpu, float>( CpuFloatData(), "MLP: cpu_mlp_float" );
    }

    TEST_F( MLPTests, Cpu_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cpu, float>( CpuFloatData(), false );
    }

    TEST_F( MLPTests, Cpu_Float_DeviceType ) {
        TestDeviceType<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_SubModules ) {
        TestSubModules<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_SaveLoad ) {
        TestSaveLoad<DeviceType::Cpu, float>( CpuFloatData() );
    }

    TEST_F( MLPTests, NoBias_Cpu_Float_ParameterCount ) {
        TestParameterCount<DeviceType::Cpu, float>( NoBiasCpuFloatData() );
    }

    TEST_F( MLPTests, NoBias_Cpu_Float_TestForward ) {
        TestForward<DeviceType::Cpu, float>( NoBiasCpuFloatData() );
    }

    TEST_F( MLPTests, LayerNorm_Cpu_Float_TestForward ) {
        TestForward<DeviceType::Cpu, float>( LayerNormCpuFloatData() );
    }

    TEST_F( MLPTests, LayerNorm_Cpu_Float_SubModules ) {
        TestSubModules<DeviceType::Cpu, float>( LayerNormCpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Training_Float_TrainingMode ) {
        TestTrainingMode<DeviceType::Cpu, float>( TrainingCpuFloatData(), true );
    }

    TEST_F( MLPTests, Cpu_Training_Float_TestForward ) {
        TestForward<DeviceType::Cpu, float>( TrainingCpuFloatData() );
    }

    TEST_F( MLPTests, Cuda_Float_TestName ) {
        try {
            TestGetName<DeviceType::Cuda, float>( CudaFloatData(), "cuda_mlp_float" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_ParameterCount ) {
        try {
            TestParameterCount<DeviceType::Cuda, float>( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_TestForward ) {
        try {
            TestForward<DeviceType::Cuda, float>( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_TestToString ) {
        try {
            TestToString<DeviceType::Cuda, float>( CudaFloatData(), "MLP: cuda_mlp_float" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_TrainingMode ) {
        try {
            TestTrainingMode<DeviceType::Cuda, float>( CudaFloatData(), false );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_DeviceType ) {
        try {
            TestDeviceType<DeviceType::Cuda, float>( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Float_SubModules ) {
        try {
            TestSubModules<DeviceType::Cuda, float>( CudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, NoBias_Cuda_Float_ParameterCount ) {
        try {
            TestParameterCount<DeviceType::Cuda, float>( NoBiasCudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, NoBias_Cuda_Float_TestForward ) {
        try {
            TestForward<DeviceType::Cuda, float>( NoBiasCudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, LayerNorm_Cuda_Float_TestForward ) {
        try {
            TestForward<DeviceType::Cuda, float>( LayerNormCudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, LayerNorm_Cuda_Float_SubModules ) {
        try {
            TestSubModules<DeviceType::Cuda, float>( LayerNormCudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Training_Float_TrainingMode ) {
        try {
            TestTrainingMode<DeviceType::Cuda, float>( TrainingCudaFloatData(), true );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Training_Float_TestForward ) {
        try {
            TestForward<DeviceType::Cuda, float>( TrainingCudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Half_TestName ) {
        try {
            TestGetName<DeviceType::Cuda, half>( CudaHalfData(), "cuda_mlp_half" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Half_ParameterCount ) {
        try {
            TestParameterCount<DeviceType::Cuda, half>( CudaHalfData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Half_TestForward ) {
        try {
            TestForward<DeviceType::Cuda, half>( CudaHalfData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Half_TestToString ) {
        try {
            TestToString<DeviceType::Cuda, half>( CudaHalfData(), "MLP: cuda_mlp_half" );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_Half_TrainingMode ) {
        try {
            TestTrainingMode<DeviceType::Cuda, half>( CudaHalfData(), false );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_PerformancePrecision_Policy ) {
        try {
            TestForward<DeviceType::Cuda, float>( PerfPrecisionCudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_AccuracyPrecision_Policy ) {
        try {
            TestForward<DeviceType::Cuda, float>( AccuracyPrecisionCudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Cuda_NativePrecision_Policy ) {
        try {
            TestForward<DeviceType::Cuda, float>( NativePrecisionCudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Context_Cpu_Float_DeviceType ) {
        TestDeviceType<DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    TEST_F( MLPTests, Context_Cpu_Float_Forward ) {
        TestForward<DeviceType::Cpu, float>( ContextCpuFloatData() );
    }

    TEST_F( MLPTests, Cpu_Float_EdgeCases ) {
        TestEdgeCases<DeviceType::Cpu, float>();
    }

    TEST_F( MLPTests, Cuda_Float_EdgeCases ) {
        try {
            TestEdgeCases<DeviceType::Cuda, float>();
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, CpuCuda_Forward_Output_Equivalence ) {
        try {
            TestCpuCudaEquivalence<float>( CpuFloatData(), CudaFloatData() );
        }
        catch ( const std::exception& ) {
            GTEST_SKIP() << "CUDA device not available, skipping test";
        }
    }

    TEST_F( MLPTests, Constructor_InvalidConfiguration ) {
        MLPConfig invalid_config( 0, 1024 );

        EXPECT_THROW(
            (MLP<DeviceType::Cpu, float>( "CPU", invalid_config )),
            std::invalid_argument
        );
    }

    TEST_F( MLPTests, Constructor_DeviceNameValidation ) {
        MLPConfig config( { 2, 16, 768 }, 3072 );
        config.withName( "validation_test" );

        EXPECT_NO_THROW( (MLP<DeviceType::Cpu, float>( "CPU", config )) );
    }

    TEST_F( MLPTests, Constructor_DeviceContextValidation ) {
        MLPConfig config( { 2, 16, 768 }, 3072 );
        config.withName( "context_validation_test" );

        auto cpu_context = std::make_shared<DeviceContext>( "CPU" );
        EXPECT_NO_THROW( (MLP<DeviceType::Cpu, float>( cpu_context, config )) );
    }
}