/**
 * @file Llama32Reference.h
 * @brief Llama 3.2 3B Instruct against HuggingFace's FP32 run on its BF16 weights, for any device, precision, weight
 *        format and cache format.
 *
 * The capture is Lib/Tools/Converters/Llama/hf_llama32_3b_reference.py, and the checks are HuggingFaceReference.h's. CUDA's
 * run of each format sets the bars Metal is held to (Notebooks/MetalFirstModel.md, Notebooks/MetalQuantizedWeights.md).
 * Include after `import Mila;`.
 */

#pragma once

#include "../HuggingFaceReference.h"

namespace Mila::Tests::Dnn::Models::Llama32
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    using HuggingFaceReference::GreedyAgreement;
    using HuggingFaceReference::LayerRange;
    using HuggingFaceReference::LogitsResult;
    using HuggingFaceReference::SequenceResult;
    using HuggingFaceReference::StepDistance;
    using HuggingFaceReference::distance;
    using HuggingFaceReference::precisionName;
    using HuggingFaceReference::runName;

    namespace fs = std::filesystem;

    inline constexpr dim_t kContext = 1024;

    inline fs::path weightsPath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_3b_instruct_bf16.bin";
    }

    /// The published FP4 package's weights, and FP8 weights `ExportArtifact --quantization fp8` wrote from the BF16 ones.
    inline fs::path fp4WeightsPath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_3b_instruct_fp4.safetensors";
    }

    inline fs::path fp8WeightsPath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_3b_instruct_fp8.safetensors";
    }

    /// The runtime setting a load is given for a weight policy, and for a cache policy.
    template<typename TWeightPolicy>
    constexpr WeightQuantization weightQuantizationOf()
    {
        if constexpr ( std::is_same_v<TWeightPolicy, PerChannelFp8<>> )
        {
            return WeightQuantization::FP8;
        }
        else if constexpr ( std::is_same_v<TWeightPolicy, PerGroupFp4<128>> )
        {
            return WeightQuantization::FP4;
        } else {
            static_assert( std::is_same_v<TWeightPolicy, NoWeightQuant> );

            return WeightQuantization::None;
        }
    }

    template<typename TKvPolicy>
    constexpr KvCacheCompression kvCacheCompressionOf()
    {
        return std::is_same_v<TKvPolicy, PerTokenKvFp8<>> ? KvCacheCompression::FP8 : KvCacheCompression::None;
    }

    inline fs::path referencePath()
    {
        return fs::path( TEST_DATA_DIR ) / "models" / "llama" / "llama32_3b_instruct_reference.safetensors";
    }

    inline bool captureExists()
    {
        return fs::exists( weightsPath() ) && fs::exists( referencePath() );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = NoWeightQuant, typename TKvPolicy = NoKvCompression>
    using Network = LlamaTransformer<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>;

    template<DeviceType TDeviceType, TensorDataType TPrecision>
    LlamaConfig capturedConfig( const fs::path& weights = weightsPath() )
    {
        Serialization::WeightsReader reader( weights );

        return LlamaModel<TDeviceType, TPrecision>::configFromMetadata( reader.getWeightsMetadata() );
    }

    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = NoWeightQuant, typename TKvPolicy = NoKvCompression>
    std::unique_ptr<Network<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>> loadedNetwork( DeviceId device, const fs::path& weights )
    {
        return Measurement::buildMeasuredNetwork<Network<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>>(
            weights, capturedConfig<TDeviceType, TPrecision>( weights ), device, kContext );
    }

    /// The capture and the network, as HuggingFaceReference.h's checks take a family.
    struct Family
    {
        static fs::path referencePath() { return Llama32::referencePath(); }

        /// The FP32 and BF16 runs are in one file.
        static fs::path bf16ReferencePath() { return Llama32::referencePath(); }

        static double softcap( const LlamaConfig& ) { return 0.0; }

        template<DeviceType TDeviceType, TensorDataType TPrecision>
        static LlamaConfig capturedConfig( const fs::path& weights ) { return Llama32::capturedConfig<TDeviceType, TPrecision>( weights ); }

        template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy, typename TKvPolicy>
        static auto loadedNetwork( DeviceId device, const fs::path& weights )
        {
            return Llama32::loadedNetwork<TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights );
        }
    };

    template<typename TValue>
    std::vector<TValue> referenceValues( const std::string& name )
    {
        return HuggingFaceReference::referenceValues<Family, TValue>( name );
    }

    /// Prefill the chat prompt, then decode HuggingFace's own greedy continuation, teacher-forced.
    /// `position_offset` decodes each token one position away from where it is, which the gate must catch.
    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = NoWeightQuant, typename TKvPolicy = NoKvCompression>
    LogitsResult measureLogits( DeviceId device, dim_t position_offset = 0, const fs::path& weights = weightsPath() )
    {
        return HuggingFaceReference::measureLogits<Family, TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, position_offset, weights );
    }

    /// The mean negative log-likelihood over the 1024-token wikitext sequence, with every block's output range observed.
    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = NoWeightQuant, typename TKvPolicy = NoKvCompression>
    SequenceResult measureSequence( DeviceId device, const fs::path& weights = weightsPath() )
    {
        return HuggingFaceReference::measureSequence<Family, TDeviceType, TPrecision, TWeightPolicy, TKvPolicy>( device, weights );
    }

    /// Greedy generation through the model's own load and loop, which plans the deployment as an application's would.
    template<DeviceType TDeviceType, TensorDataType TPrecision, typename TWeightPolicy = NoWeightQuant, typename TKvPolicy = NoKvCompression>
    std::vector<std::int32_t> generateGreedy( DeviceId device, int max_new_tokens, const fs::path& weights = weightsPath() )
    {
        LlamaModelConfig config( kContext );
        config.withWeightQuantization( weightQuantizationOf<TWeightPolicy>() ).withKvCacheCompression( kvCacheCompressionOf<TKvPolicy>() );

        auto model = LlamaModel<TDeviceType, TPrecision>::load( weights, config, device );

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
