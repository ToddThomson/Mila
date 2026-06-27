/**
 * @file TopKDecoder.ixx
 * @brief TopK decoder implementation for language models.
 */

module;
#include <memory>
#include <type_traits>
#include <stdexcept>
#include <optional>

export module Dnn.Decoders.TopKDecoder;
import :Config;

import Dnn.DecoderBase;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorTypes;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief TopK decoder for language models.
     *
     * Implements a top-k sampling strategy for decoding tokens from a language model.
     * The decoder selects the top-k tokens based on their probabilities and samples
     * from this reduced set to generate the next token in the sequence.
     *
     * @tparam TDeviceType The device type (e.g., CPU, CUDA) for tensor operations.
     * @tparam TPrecision  The precision type (e.g., FP32, BF16) for tensor operations.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class TopKDecoder : public Decoder<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        /**
         * @brief Constructs a TopKDecoder with the specified configuration.
         *
         * @param name The name of the decoder.
         * @param config The configuration for the TopK decoder.
         * @param device_id Optional device identifier. If provided, the decoder will create and own an ExecutionContext for the specified device.
         */
        explicit TopKDecoder( const std::string& name, const TopKConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : config_( config )
        {
            config_.validate();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                    throw std::invalid_argument( "TopKDecoder: device type mismatch" );

                // FIXME: context_ = createExecutionContext( device_id.value() );

                // FIXME: this->setExecutionContext( owned_exec_context_.get() );
            }
        }
        
        /**
         * @brief Decodes the next token based on the provided logits tensor.
         *
         * @param logits A tensor containing the logits for the current step.
         * @return The index of the selected token.
         */
        int decode( const ITensor logits )
        {
            // Implementation of top-k sampling logic goes here
            // This is a placeholder for demonstration purposes
            return 0; // Placeholder return value
        }

    private:

        TopKConfig config_; ///< Configuration for the TopK decoder.
        IExecutionContext* context_;
        size_t k_; ///< The number of top tokens to consider during decoding.
    };
}
