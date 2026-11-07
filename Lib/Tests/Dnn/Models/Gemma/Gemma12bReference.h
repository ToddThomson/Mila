/**
 * @file Gemma12bReference.h
 * @brief Gemma 4 12B Instruct in Q4_0 against HuggingFace's FP32 run on the package's own weights, decoded.
 *
 * The capture is Lib/Tools/Converters/Gemma/gemma_4_BF16/hf_gemma4_12b_q4_0_reference.py, and the checks are
 * HuggingFaceReference.h's (Notebooks/MetalGemma.md section 1). Include after `import Mila;`.
 */

#pragma once

#include "../HuggingFaceReference.h"

namespace Mila::Tests::Dnn::Models::Gemma12b
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    using HuggingFaceReference::GreedyAgreement;
    using HuggingFaceReference::LayerRange;
    using HuggingFaceReference::LogitsResult;
    using HuggingFaceReference::PiecesResult;
    using HuggingFaceReference::SequenceResult;
    using HuggingFaceReference::runName;

    namespace fs = std::filesystem;

    /// The captured sequence's length, past the local layers' window of 1024.
    inline constexpr dim_t kContext = 2048;

    /// The capture's prefill pieces and its first row either side of the window's edge, as the script writes them.
    inline constexpr dim_t kPieceTokens = 32;
    inline constexpr dim_t kWindowEdgeFirstRow = 992;

    inline fs::path weightsPath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_12b_it_qat_q4_0.safetensors";
    }

    inline fs::path referencePath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_12b_it_q4_0_reference.safetensors";
    }

    inline fs::path bf16ReferencePath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "gemma" / "gemma4_12b_it_q4_0_reference_bf16.safetensors";
    }

    inline bool captureExists()
    {
        return fs::exists( weightsPath() ) && fs::exists( referencePath() ) && fs::exists( bf16ReferencePath() );
    }

    /// The local layers keep the ring GemmaModel builds; `TKvPolicy` is the global layers' cache.
    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = PerGroupInt4<32>, typename TKvPolicy = NoKvCompression>
    using Network = GemmaTransformer<TDeviceType, TPrecision, TWeightPolicy,
        typename GemmaModel<TDeviceType, TPrecision>::GemmaSlidingKvPolicy, GemmaFeedForward::Dense, TKvPolicy>;

    template<DeviceType TDeviceType, TensorDataType TPrecision>
    GemmaConfig capturedConfig( const fs::path& weights = weightsPath() )
    {
        Serialization::WeightsReader reader( weights );

        return GemmaModel<TDeviceType, TPrecision>::configFromMetadata( reader.getWeightsMetadata() );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = PerGroupInt4<32>, typename TKvPolicy = NoKvCompression>
    std::unique_ptr<Network<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>> loadedNetwork( DeviceId device, const fs::path& weights )
    {
        return Measurement::buildMeasuredNetwork<Network<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>>(
            weights, capturedConfig<TDeviceType, TPrecision>( weights ), device, kContext );
    }

    struct Family
    {
        static fs::path referencePath() { return Gemma12b::referencePath(); }
        static fs::path bf16ReferencePath() { return Gemma12b::bf16ReferencePath(); }
        static double softcap( const GemmaConfig& config ) { return config.getFinalLogitSoftcapping(); }

        template<DeviceType TDeviceType, TensorDataType TPrecision>
        static GemmaConfig capturedConfig( const fs::path& weights ) { return Gemma12b::capturedConfig<TDeviceType, TPrecision>( weights ); }

        template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy, typename TKvPolicy>
        static auto loadedNetwork( DeviceId device, const fs::path& weights )
        {
            return Gemma12b::loadedNetwork<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights );
        }
    };

    template<typename TValue>
    std::vector<TValue> referenceValues( const std::string& name )
    {
        return HuggingFaceReference::referenceValues<Family, TValue>( name );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = PerGroupInt4<32>, typename TKvPolicy = NoKvCompression>
    LogitsResult measureLogits( DeviceId device, dim_t position_offset = 0, const fs::path& weights = weightsPath() )
    {
        return HuggingFaceReference::measureLogits<Family, TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, position_offset, weights );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = PerGroupInt4<32>, typename TKvPolicy = NoKvCompression>
    SequenceResult measureSequence( DeviceId device, const fs::path& weights = weightsPath() )
    {
        return HuggingFaceReference::measureSequence<Family, TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = PerGroupInt4<32>, typename TKvPolicy = NoKvCompression>
    PiecesResult measurePieces( DeviceId device, const fs::path& weights = weightsPath() )
    {
        return HuggingFaceReference::measurePieces<Family, TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights, kPieceTokens );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = PerGroupInt4<32>, typename TKvPolicy = NoKvCompression>
    LogitsResult measureWindowEdge( DeviceId device, const fs::path& weights = weightsPath() )
    {
        return HuggingFaceReference::measureWindowEdge<Family, TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights, kWindowEdgeFirstRow );
    }

    /// Greedy generation through the model's own load and loop, which plans the deployment as an application's would.
    template<DeviceType TDeviceType, TensorDataType TPrecision>
    std::vector<std::int32_t> generateGreedy( DeviceId device, int max_new_tokens, const fs::path& weights = weightsPath() )
    {
        GemmaModelConfig config( kContext );
        config.withWeightQuantization( WeightQuantization::Q4_0 );

        auto model = GemmaModel<TDeviceType, TPrecision>::load( weights, config, device );

        GenerateParams params;
        params.max_new_tokens = max_new_tokens;
        params.sampling.temperature = 0.0f;
        params.sampling.top_k = 0;

        std::vector<std::int32_t> tokens;

        (void)model->generate( referenceValues<std::int32_t>( "prompt" ),
            [&]( std::int32_t token ) { tokens.push_back( token ); }, params, std::stop_token{} );

        return tokens;
    }

    inline GreedyAgreement greedyAgreement( TensorDataType precision, const std::vector<std::int32_t>& generated,
        const std::vector<double>& largest_error, std::int64_t vocab )
    {
        return HuggingFaceReference::greedyAgreement<Family>( precision, generated, largest_error, vocab );
    }
}
