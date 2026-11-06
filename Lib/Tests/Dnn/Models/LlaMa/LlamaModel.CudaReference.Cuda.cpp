/**
 * @file LlamaModel.CudaReference.Cuda.cpp
 * @brief Mila's CUDA run of Llama 3.2 3B and Llama 3.1 8B on their FP4 packages, captured as the reference Metal's run of the
 *        same weights is compared with.
 *
 * Writes each capture beside its weights (Notebooks/MetalQuantizedWeights.md section 1.2). Needs the weights and the 3B's
 * HuggingFace capture, whose prompt and sequence it reuses, so it never runs in CI.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <stop_token>
#include <string>
#include <type_traits>
#include <vector>

import Mila;

#include "Llama32Reference.h"
#include "LlamaCudaReference.h"

namespace Mila::Tests::Dnn::Models::Llama32
{
    namespace
    {
        void writeCapture( const fs::path& weights )
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }

            if ( !fs::exists( weights ) || !fs::exists( referencePath() ) )
            {
                GTEST_SKIP() << "Not present: " << weights.string() << " or " << referencePath().string();
            }

            const fs::path capture = writeCudaCapture<PerGroupFp4<128>>( weights, Device::Cuda( 0 ) );

            EXPECT_TRUE( fs::exists( capture ) );
        }
    }

    TEST( LlamaModelCudaReference, Llama32Fp4_WritesTheCapture )
    {
        writeCapture( fp4WeightsPath() );
    }

    TEST( LlamaModelCudaReference, Llama31Fp4_WritesTheCapture )
    {
        writeCapture( llama31Fp4WeightsPath() );
    }
}
