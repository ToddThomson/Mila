/**
 * @file LlamaModel.CudaReference.Metal.cpp
 * @brief Llama 3.2 3B and Llama 3.1 8B on Metal on their FP4 packages, against Mila's CUDA run of the same weights.
 *
 * The 8B has no HuggingFace capture, so this is its reference; the bar on its distance is not decided
 * (Notebooks/MetalQuantizedWeights.md section 2.3). Needs the weights and LlamaModel.CudaReference.Cuda.cpp's captures.
 */

#include <gtest/gtest.h>

import Mila;
import std;

#include "Llama32Reference.h"
#include "LlamaCudaReference.h"

namespace Mila::Tests::Dnn::Models::Llama32
{
    namespace
    {
        template<typename TPrecisionConstant>
        class LlamaModelCudaReferenceMetal : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                if ( getDeviceCount( DeviceType::Metal ) == 0 )
                {
                    GTEST_SKIP() << "No Metal device available";
                }
            }
        };

        using Precisions = ::testing::Types<
            std::integral_constant<TensorDataType, TensorDataType::FP16>,
            std::integral_constant<TensorDataType, TensorDataType::BF16>>;

        struct PrecisionNames
        {
            template<typename T>
            static std::string GetName( int )
            {
                return T::value == TensorDataType::FP16 ? "Fp16" : "Bf16";
            }
        };

        TYPED_TEST_SUITE( LlamaModelCudaReferenceMetal, Precisions, PrecisionNames );

        bool present( const fs::path& weights )
        {
            return fs::exists( weights ) && fs::exists( cudaCapturePath( weights ) );
        }

        void expectFiniteAndAtTies( const CaptureComparison& result )
        {
            EXPECT_EQ( result.non_finite, 0 );
            EXPECT_TRUE( result.argmax_disagreements.empty() ) << "first at step " << result.argmax_disagreements.front();
            EXPECT_TRUE( result.greedy.diverged_at_a_tie ) << "diverged after " << result.greedy.agreed << " tokens";
        }
    }

    // Each argmax and the greedy run may differ from CUDA's only at a tie; the distances are reported.
    TYPED_TEST( LlamaModelCudaReferenceMetal, Llama32Fp4 )
    {
        if ( !present( fp4WeightsPath() ) )
        {
            GTEST_SKIP() << "Not present: " << fp4WeightsPath().string() << " or its CUDA capture";
        }

        expectFiniteAndAtTies( compareWithCudaCapture<DeviceType::Metal, TypeParam::value, PerGroupFp4<128>>( fp4WeightsPath(), Device::Metal( 0 ) ) );
    }

    TYPED_TEST( LlamaModelCudaReferenceMetal, Llama31Fp4 )
    {
        if ( !present( llama31Fp4WeightsPath() ) )
        {
            GTEST_SKIP() << "Not present: " << llama31Fp4WeightsPath().string() << " or its CUDA capture";
        }

        // Its distance from CUDA has no bar yet: the 3B's would carry CUDA's FP8 activations (Notebooks/MetalQuantizedWeights.md 2.3).
        expectFiniteAndAtTies(
            compareWithCudaCapture<DeviceType::Metal, TypeParam::value, PerGroupFp4<128>>( llama31Fp4WeightsPath(), Device::Metal( 0 ) ) );
    }
}
