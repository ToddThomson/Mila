/**
 * @file Qwen38L4Reference.h
 * @brief Qwen 3.8's four-layer packed build, under its per-role plan, against HuggingFace's FP32 run on its weights as Mila runs them.
 *
 * The capture is Lib/Tools/Converters/Qwen/qwen38_BF16/hf_qwen38_l4_plan_reference.py, and the checks are
 * HuggingFaceReference.h's (Notebooks/MetalQwen.md section 1.2). Include after `import Mila;`.
 */

#pragma once

#include "../HuggingFaceReference.h"

namespace Mila::Tests::Dnn::Models::Qwen38L4
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::KvCache;

    using HuggingFaceReference::GreedyAgreement;
    using HuggingFaceReference::LayerRange;
    using HuggingFaceReference::LogitsResult;
    using HuggingFaceReference::PiecesResult;
    using HuggingFaceReference::SequenceResult;

    namespace fs = std::filesystem;

    /// The captured sequence's length, and its prefill pieces, as the script writes them.
    inline constexpr dim_t kContext = 2048;
    inline constexpr dim_t kPieceTokens = 32;

    inline fs::path weightsPath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_l4_2p9bit.safetensors";
    }

    inline fs::path referencePath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_l4_2p9bit_reference.safetensors";
    }

    inline fs::path bf16ReferencePath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "qwen" / "qwen38_27b_l4_2p9bit_reference_bf16.safetensors";
    }

    inline bool captureExists()
    {
        return fs::exists( weightsPath() ) && fs::exists( referencePath() ) && fs::exists( bf16ReferencePath() );
    }

    /// The network the package's plan builds, `TWeightPlan` the plan rather than one format.
    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPlan = QwenPrecisionPlan, typename TKvPolicy = NoKvCompression>
    using Network = QwenTransformer<TDeviceType, TPrecision, TWeightPlan, TKvPolicy>;

    template<DeviceType TDeviceType, TensorDataType TPrecision>
    QwenConfig capturedConfig( const fs::path& weights = weightsPath() )
    {
        Serialization::WeightsReader reader( weights );

        return QwenModel<TDeviceType, TPrecision>::configFromMetadata( reader.getWeightsMetadata() );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPlan = QwenPrecisionPlan, typename TKvPolicy = NoKvCompression>
    std::unique_ptr<Network<TDeviceType, TPrecision, TWeightPlan, TKvPolicy>> loadedNetwork( DeviceId device, const fs::path& weights )
    {
        return Measurement::buildMeasuredNetwork<Network<TDeviceType, TPrecision, TWeightPlan, TKvPolicy>>(
            weights, capturedConfig<TDeviceType, TPrecision>( weights ), device, kContext );
    }

    struct Family
    {
        static fs::path referencePath() { return Qwen38L4::referencePath(); }
        static fs::path bf16ReferencePath() { return Qwen38L4::bf16ReferencePath(); }
        static double softcap( const QwenConfig& ) { return 0.0; }

        template<DeviceType TDeviceType, TensorDataType TPrecision>
        static QwenConfig capturedConfig( const fs::path& weights ) { return Qwen38L4::capturedConfig<TDeviceType, TPrecision>( weights ); }

        template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPlan, typename TKvPolicy>
        static auto loadedNetwork( DeviceId device, const fs::path& weights )
        {
            return Qwen38L4::loadedNetwork<TDeviceType, TPrecision, TWeightPlan, TKvPolicy>( device, weights );
        }
    };

    template<DeviceType TDeviceType, TensorDataType TPrecision>
    LogitsResult measureLogits( DeviceId device, dim_t position_offset = 0 )
    {
        return HuggingFaceReference::measureLogits<Family, TDeviceType, TPrecision, QwenPrecisionPlan, NoKvCompression>(
            device, position_offset, weightsPath() );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision>
    SequenceResult measureSequence( DeviceId device )
    {
        return HuggingFaceReference::measureSequence<Family, TDeviceType, TPrecision, QwenPrecisionPlan, NoKvCompression>( device, weightsPath() );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision>
    PiecesResult measurePieces( DeviceId device )
    {
        return HuggingFaceReference::measurePieces<Family, TDeviceType, TPrecision, QwenPrecisionPlan, NoKvCompression>(
            device, weightsPath(), kPieceTokens );
    }

    /// Greedy generation through the model's own load and loop, under the package's plan.
    template<DeviceType TDeviceType, TensorDataType TPrecision>
    std::vector<std::int32_t> generateGreedy( DeviceId device, int max_new_tokens )
    {
        QwenModelConfig config( kContext );
        config.withPrecisionPlan();

        auto model = QwenModel<TDeviceType, TPrecision>::load( weightsPath(), config, device );

        GenerateParams params;
        params.max_new_tokens = max_new_tokens;
        params.sampling.temperature = 0.0f;
        params.sampling.top_k = 0;

        std::vector<std::int32_t> tokens;

        (void)model->generate( HuggingFaceReference::referenceValues<Family, std::int32_t>( "prompt" ),
            [&]( std::int32_t token ) { tokens.push_back( token ); }, params, std::stop_token{} );

        return tokens;
    }

    inline GreedyAgreement greedyAgreement( TensorDataType precision, const std::vector<std::int32_t>& generated,
        const std::vector<double>& largest_error, std::int64_t vocab )
    {
        return HuggingFaceReference::greedyAgreement<Family>( precision, generated, largest_error, vocab );
    }
}
