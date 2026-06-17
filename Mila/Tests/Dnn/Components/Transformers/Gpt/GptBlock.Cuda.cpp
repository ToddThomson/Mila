/**
 * @file GptBlock.Cuda.cpp
 * @brief Concrete-component tests for GptBlock<DeviceType::Cuda, FP32>.
 *
 * The CUDA mirror of GptBlock.Cpu.cpp. GptBlock is the GPT-2 pre-LN transformer
 * block composite:
 *   LayerNorm -> QKV(Linear) -> MultiHeadAttention -> out(Linear) -> Residual ->
 *   LayerNorm -> MLP -> Residual
 *
 * Per the concrete-component archetype (Specifications/Testing.md) this tests the
 * GptBlock DELTA over the Component / CompositeComponent base: construction/
 * validation, the build lifecycle and shape contract, the RuntimeMode/TrainingMode
 * axis, forward shape/finiteness, and that gradients FLOW through the whole stack.
 * The leaf numerics are proven in their own suites; here the point is that the
 * composition wires forward and backward together correctly. The GPT-2 CUDA
 * forward/backward kernels are exercised by the Bard training sample, so finiteness
 * and a non-zero composed input gradient are asserted here.
 *
 * The numeric finite-difference gradient probe from the CPU sibling is intentionally
 * NOT mirrored here: it belongs to the net-new gradient-check archetype (BACKLOG),
 * and end-to-end gradient correctness is covered by the GptTransformer oracle.
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Gpt
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using GptBlockCuda = Mila::Dnn::GptBlock<DeviceType::Cuda, TensorDataType::FP32>;
        using TensorCuda = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using TensorHost = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        void fillSpread( TensorHost& t )
        {
            auto* data = t.data();

            for ( size_t i = 0; i < t.size(); ++i )
            {
                data[ i ] = static_cast<float>( i ) / t.size() * 2.0f - 1.0f;
            }
        }

        bool allFinite( const TensorHost& t )
        {
            const auto* data = t.data();

            for ( size_t i = 0; i < t.size(); ++i )
            {
                if ( !std::isfinite( data[ i ] ) )
                {
                    return false;
                }
            }

            return true;
        }

        bool anyNonZero( const TensorHost& t )
        {
            const auto* data = t.data();

            for ( size_t i = 0; i < t.size(); ++i )
            {
                if ( data[ i ] != 0.0f )
                {
                    return true;
                }
            }

            return false;
        }

        static_assert( GptBlockCuda::getDeviceType() == DeviceType::Cuda );
        static_assert( GptBlockCuda::getPrecision() == TensorDataType::FP32 );
    }

    class GptBlockCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }

        // Tiny block, parameters initialized (BuildContext default), so forward
        // produces finite values and the gradient flow has real weights.
        std::unique_ptr<GptBlockCuda> builtBlock(
            int64_t model_dim, int64_t num_heads, int64_t hidden, RuntimeMode mode )
        {
            GptBlockConfig config( model_dim, num_heads );
            config.withHiddenSize( hidden );

            auto block = std::make_unique<GptBlockCuda>( "gpt_block", config, Device::Cuda( 0 ) );
            block->build( BuildContext( shape_t{ batch_, seq_, model_dim }, mode ) );

            return block;
        }

        // Spread-filled device tensor (host-staged then copied).
        TensorCuda spreadInput( const shape_t& shape )
        {
            TensorHost host( Device::Cpu(), shape );
            fillSpread( host );

            TensorCuda device( Device::Cuda( 0 ), shape );
            copy( host, device );

            return device;
        }

        // Device -> host copy for value assertions.
        TensorHost toHost( const TensorCuda& device, GptBlockCuda& block )
        {
            TensorHost host( Device::Cpu(), device.shape() );
            copy( device, host );
            block.synchronize();

            return host;
        }

        static constexpr int64_t batch_ = 1;
        static constexpr int64_t seq_ = 3;
        static constexpr int64_t model_dim_ = 4;
        static constexpr int64_t num_heads_ = 2;
        static constexpr int64_t hidden_ = 8;
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( GptBlockCudaTests, Construct_StandaloneSucceeds )
    {
        GptBlockConfig config( 64, 8 );
        GptBlockCuda block( "gpt_block", config, Device::Cuda( 0 ) );

        EXPECT_EQ( block.getName(), "gpt_block" );
        EXPECT_EQ( block.getDeviceId().type, DeviceType::Cuda );
    }

    TEST_F( GptBlockCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        GptBlockConfig config( 64, 8 );

        EXPECT_THROW( GptBlockCuda( "gpt_block", config, Device::Cpu() ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( GptBlockCudaTests, Build_SetsIsBuilt )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Inference );

        EXPECT_TRUE( block->isBuilt() );
    }

    TEST_F( GptBlockCudaTests, Build_ThrowsOnNonRank3Input )
    {
        GptBlockConfig config( model_dim_, num_heads_ );
        GptBlockCuda block( "gpt_block", config, Device::Cuda( 0 ) );

        // Rank-2 input violates the [B, T, embedding_dim] contract.
        EXPECT_THROW( block.build( BuildContext( shape_t{ seq_, model_dim_ }, RuntimeMode::Inference ) ),
            std::invalid_argument );
    }

    TEST_F( GptBlockCudaTests, Build_ThrowsOnModelDimMismatch )
    {
        GptBlockConfig config( model_dim_, num_heads_ );
        GptBlockCuda block( "gpt_block", config, Device::Cuda( 0 ) );

        // Last dim (model_dim + 1) does not match the configured embedding dim.
        EXPECT_THROW( block.build( BuildContext( shape_t{ batch_, seq_, model_dim_ + 1 }, RuntimeMode::Inference ) ),
            std::invalid_argument );
    }

    TEST_F( GptBlockCudaTests, Forward_ThrowsBeforeBuild )
    {
        GptBlockConfig config( model_dim_, num_heads_ );
        GptBlockCuda block( "gpt_block", config, Device::Cuda( 0 ) );
        TensorCuda input( Device::Cuda( 0 ), shape_t{ batch_, seq_, model_dim_ } );

        EXPECT_THROW( block.forward( input ), std::runtime_error );
    }

    TEST_F( GptBlockCudaTests, Backward_ThrowsBeforeBuild )
    {
        GptBlockConfig config( model_dim_, num_heads_ );
        GptBlockCuda block( "gpt_block", config, Device::Cuda( 0 ) );
        TensorCuda input( Device::Cuda( 0 ), shape_t{ batch_, seq_, model_dim_ } );
        TensorCuda output_grad( Device::Cuda( 0 ), shape_t{ batch_, seq_, model_dim_ } );

        EXPECT_THROW( block.backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // C. Execution Context / Device
    // ====================================================================

    TEST_F( GptBlockCudaTests, Synchronize_Succeeds )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Inference );

        EXPECT_NO_THROW( block->synchronize() );
    }

    // ====================================================================
    // D. Runtime + Training Mode
    // ====================================================================

    TEST_F( GptBlockCudaTests, BuiltForTraining_IsTrainingMode )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Training );

        EXPECT_TRUE( block->isTrainingMode() );
        EXPECT_FALSE( block->isInferenceMode() );
    }

    TEST_F( GptBlockCudaTests, BuiltForInference_IsInferenceMode )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Inference );

        EXPECT_TRUE( block->isInferenceMode() );
        EXPECT_FALSE( block->isTrainingMode() );
    }

    TEST_F( GptBlockCudaTests, Backward_ThrowsWhenBuiltForInference )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Inference );

        auto input = spreadInput( shape_t{ batch_, seq_, model_dim_ } );
        TensorCuda output_grad( Device::Cuda( 0 ), shape_t{ batch_, seq_, model_dim_ } );

        block->forward( input );

        EXPECT_THROW( block->backward( input, output_grad ), std::runtime_error );
    }

    TEST_F( GptBlockCudaTests, Backward_ThrowsWhenForwardNotCalled )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Training );

        auto input = spreadInput( shape_t{ batch_, seq_, model_dim_ } );
        TensorCuda output_grad( Device::Cuda( 0 ), shape_t{ batch_, seq_, model_dim_ } );

        EXPECT_THROW( block->backward( input, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // E. Forward (shape + finiteness)
    // ====================================================================

    TEST_F( GptBlockCudaTests, Forward_PreservesShape )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Inference );

        auto input = spreadInput( shape_t{ batch_, seq_, model_dim_ } );
        auto& output = block->forward( input );

        EXPECT_EQ( output.shape(), ( shape_t{ batch_, seq_, model_dim_ } ) );
    }

    TEST_F( GptBlockCudaTests, Forward_ProducesFiniteOutput )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Training );

        auto input = spreadInput( shape_t{ batch_, seq_, model_dim_ } );
        auto& output = block->forward( input );

        EXPECT_TRUE( allFinite( toHost( output, *block ) ) );
    }

    // ====================================================================
    // F. Backward (gradient flows through the stack)
    // ====================================================================

    TEST_F( GptBlockCudaTests, Backward_ReturnsFiniteNonZeroInputGradient )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Training );

        const shape_t shape{ batch_, seq_, model_dim_ };
        auto input = spreadInput( shape );

        TensorHost host_grad( Device::Cpu(), shape );
        for ( size_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 0.1f * static_cast<float>( i + 1 );
        }

        TensorCuda output_grad( Device::Cuda( 0 ), shape );
        copy( host_grad, output_grad );

        block->forward( input );
        auto& input_grad = block->backward( input, output_grad );

        EXPECT_EQ( input_grad.shape(), shape );

        const TensorHost host_input_grad = toHost( input_grad, *block );
        EXPECT_TRUE( allFinite( host_input_grad ) );
        EXPECT_TRUE( anyNonZero( host_input_grad ) ) << "expected non-zero gradient flowing through the stack";
    }

    // ====================================================================
    // G. Parameters & Components
    // ====================================================================

    TEST_F( GptBlockCudaTests, GetComponents_ReturnsEightChildren )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Inference );

        // attn, ln1, ln2, qkv_proj, out_proj, res1, res2, mlp.
        EXPECT_EQ( block->getComponents().size(), 8u );
    }

    TEST_F( GptBlockCudaTests, ParameterCount_Positive )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Inference );

        EXPECT_GT( block->parameterCount(), 0u );
    }

    TEST_F( GptBlockCudaTests, GetGradients_PresentWhenBuiltForTraining )
    {
        auto block = builtBlock( model_dim_, num_heads_, hidden_, RuntimeMode::Training );

        EXPECT_FALSE( block->getGradients().empty() );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( GptBlockCudaTests, ToString_DescribesComponent )
    {
        GptBlockConfig config( 64, 8 );
        GptBlockCuda block( "my_block", config, Device::Cuda( 0 ) );

        const std::string text = block.toString();

        EXPECT_NE( text.find( "GptBlock" ), std::string::npos );
        EXPECT_NE( text.find( "my_block" ), std::string::npos );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( GptBlockCudaTests, GetType_IsTransformer )
    {
        GptBlockConfig config( 64, 8 );
        GptBlockCuda block( "gpt_block", config, Device::Cuda( 0 ) );

        EXPECT_EQ( block.getType(), ComponentType::Transformer );
    }
}
