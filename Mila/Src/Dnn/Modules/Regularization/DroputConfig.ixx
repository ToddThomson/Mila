/**
 * @file DropoutConfig.ixx
 * @brief Configuration interface for the Dropout regularization module in the Mila DNN framework.
 *
 * Defines the DropoutConfig class, providing a type-safe fluent interface for configuring
 * Dropout regularization modules. Inherits from ModuleConfig CRTP base and adds Dropout-specific
 * options: dropout probability and training mode behavior.
 *
 * Exposed as part of the Dropout module via module partitions.
 */

module;
#include <stdexcept>

export module Dnn.Modules.Dropout:Config;

import Dnn.Module;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for Dropout module.
     *
     * Provides a type-safe fluent interface for configuring Dropout modules.
     */
    export class DropoutConfig : public ModuleConfig<DropoutConfig> {
    public:
        /**
         * @brief Default constructor.
         */
        DropoutConfig() = default;

        /**
         * @brief Constructor with dropout probability.
         *
         * @param probability The dropout probability (0.0 to 1.0)
         */
        explicit DropoutConfig( float probability ) : probability_( probability ) {}

        /**
         * @brief Configure the dropout probability.
         *
         * @param probability The probability of zeroing elements (0.0 to 1.0)
         * @return DropoutConfig& Reference to this for method chaining
         */
        DropoutConfig& withProbability( float probability ) {
            probability_ = probability;
            return *this;
        }

        /**
         * @brief Configure whether to apply scaling during inference.
         *
         * When true, outputs during inference will be scaled by 1/(1-p) to maintain
         * the same expected value between training and inference. When false,
         * dropout is completely disabled during inference.
         *
         * @param scale_during_inference Whether to apply scaling during inference
         * @return DropoutConfig& Reference to this for method chaining
         */
        DropoutConfig& withScalingDuringInference( bool scale_during_inference ) {
            scale_during_inference_ = scale_during_inference;
            return *this;
        }

        /**
         * @brief Configure whether to use the same dropout mask for all elements in a batch.
         *
         * @param use_same_mask_per_batch Whether to use the same mask for entire batch
         * @return DropoutConfig& Reference to this for method chaining
         */
        DropoutConfig& withSameMaskPerBatch( bool use_same_mask_per_batch ) {
            use_same_mask_per_batch_ = use_same_mask_per_batch;
            return *this;
        }

        /**
         * @brief Get the configured dropout probability.
         *
         * @return float The dropout probability
         */
        float getProbability() const { return probability_; }

        /**
         * @brief Check if scaling during inference is enabled.
         *
         * @return bool Whether scaling during inference is enabled
         */
        bool scalesDuringInference() const { return scale_during_inference_; }

        /**
         * @brief Check if the same mask is used for all elements in a batch.
         *
         * @return bool Whether the same mask is used per batch
         */
        bool usesSameMaskPerBatch() const { return use_same_mask_per_batch_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
         */
        void validate() const {
            ModuleConfig<DropoutConfig>::validate();

            if ( probability_ < 0.0f || probability_ >= 1.0f ) {
                throw std::invalid_argument( "Dropout probability must be in range [0, 1)" );
            }
        }

    private:
        float probability_{ 0.5f };                 ///< The probability of zeroing elements
        bool scale_during_inference_{ false };      ///< Whether to apply scaling during inference
        bool use_same_mask_per_batch_{ false };     ///< Whether to use the same mask for entire batch
    };
}