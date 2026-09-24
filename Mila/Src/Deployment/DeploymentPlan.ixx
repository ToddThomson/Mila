/**
 * @file DeploymentPlan.ixx
 * @brief One deployment that can be loaded: the values the planner resolved, what stopped each being larger,
 *        and what it was priced for.
 *
 * A load executes a plan and never re-derives a value it holds. See Specifications/Deployment.md sections 3.3,
 * 6 and 7.
 */

module;
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

export module Deployment.DeploymentPlan;

import Dnn.Component;
import Dnn.LanguageModelConfig;
import Dnn.RuntimeMode;
import Dnn.TensorTypes;
import Deployment.DeviceReading;
import Serialization.WeightsReader;
import Compute.DeviceId;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief A deployment the planner found against one reading of its device.
     *
     * It exists only when it can be loaded, as of its reading: on a card that drives a display, free memory can
     * still move before the load, which then fails cleanly on allocation (Deployment.md section 9).
     */
    export class DeploymentPlan
    {
    public:

        /// What stopped the context length being larger.
        enum class ContextLimit
        {
            /// The request fixed it.
            FixedByCaller,

            /// The model's trained maximum, or the request's ceiling.
            TrainedMaximum,

            /// A longer context would have reduced the prefill chunk (section 5, rule 3).
            FullPrefillChunk,

            /// A longer context fits at no chunk (section 5, rule 4).
            DeviceMemory
        };

        /// What stopped the prefill chunk being larger.
        enum class PrefillChunkLimit
        {
            /// Memory did not reduce it: it is the largest rung the context length admits.
            LargestTheContextPermits,

            /// A larger rung does not fit.
            DeviceMemory
        };

        /**
         * @brief Made by a family's planDeployment; a caller who wants particular values fixes them in the
         *        request instead, so they are checked like any other (section 7).
         */
        DeploymentPlan(
            const DeviceReading& reading,
            dim_t context_length,
            ContextLimit context_limit,
            const PrefillChunking& prefill,
            const MemoryStats& footprint,
            WeightQuantization weight_quantization,
            KvCacheCompression kv_cache_compression,
            dim_t language_model_head_positions,
            const WeightsMetadata& priced_for,
            std::string priced_stored_quantization )
            : reading_( reading ), context_length_( context_length ), context_limit_( context_limit ),
              prefill_( prefill ), footprint_( footprint ), weight_quantization_( weight_quantization ),
              kv_cache_compression_( kv_cache_compression ),
              language_model_head_positions_( language_model_head_positions ),
              priced_for_( priced_for ), priced_stored_quantization_( std::move( priced_stored_quantization ) )
        {
        }

        [[nodiscard]] DeviceId device() const noexcept
        {
            return reading_.device;
        }

        /// The reading every value here was decided against.
        [[nodiscard]] const DeviceReading& reading() const noexcept
        {
            return reading_;
        }

        [[nodiscard]] dim_t contextLength() const noexcept
        {
            return context_length_;
        }

        [[nodiscard]] ContextLimit contextLimitedBy() const noexcept
        {
            return context_limit_;
        }

        [[nodiscard]] dim_t prefillChunkRows() const noexcept
        {
            return prefill_.chunk_rows;
        }

        [[nodiscard]] PrefillChunkLimit prefillChunkLimitedBy() const noexcept
        {
            return prefill_.isMemoryConstrained()
                ? PrefillChunkLimit::DeviceMemory
                : PrefillChunkLimit::LargestTheContextPermits;
        }

        /// The chunk, and the largest rung the context admits memory aside.
        [[nodiscard]] const PrefillChunking& prefillChunking() const noexcept
        {
            return prefill_;
        }

        /// What the build allocates, category by category (Deployment.md G1).
        [[nodiscard]] const MemoryStats& footprint() const noexcept
        {
            return footprint_;
        }

        [[nodiscard]] WeightQuantization weightQuantization() const noexcept
        {
            return weight_quantization_;
        }

        [[nodiscard]] KvCacheCompression kvCacheCompression() const noexcept
        {
            return kv_cache_compression_;
        }

        [[nodiscard]] dim_t languageModelHeadPositions() const noexcept
        {
            return language_model_head_positions_;
        }

        /// The package facts pricing read: architecture and geometry. The name is not one of them.
        [[nodiscard]] const WeightsMetadata& pricedFor() const noexcept
        {
            return priced_for_;
        }

        /// The weight format the priced package stored; empty for reference weights.
        [[nodiscard]] const std::string& pricedStoredQuantization() const noexcept
        {
            return priced_stored_quantization_;
        }

        /**
         * @brief The family's model config this plan loads with.
         */
        template<typename TConfig>
        [[nodiscard]] TConfig modelConfig() const
        {
            TConfig config( context_length_ );
            config.withWeightQuantization( weight_quantization_ )
                .withKvCacheCompression( kv_cache_compression_ )
                .withLanguageModelHeadPositions( language_model_head_positions_ );

            return config;
        }

        /**
         * @brief The build context this plan builds with: its context length, chunk and the granularity it
         *        was priced at.
         */
        [[nodiscard]] BuildContext buildContext() const
        {
            return BuildContext( shape_t{ 1, context_length_ }, RuntimeMode::Inference, false )
                .withAllocationGranularity( reading_.allocation_granularity )
                .withPrefillSize( prefill_.chunk_rows );
        }

        /**
         * @brief Refuse a package this plan was not priced for, naming both.
         *
         * Two packages with the same architecture, geometry and stored format price identically, so a plan
         * is valid for either; a model name is not a fact pricing read.
         */
        void requirePricedFor(
            std::string_view caller, std::string_view weights_path,
            const WeightsMetadata& metadata, std::string_view stored_quantization ) const
        {
            if ( pricingFacts( metadata ) != pricingFacts( priced_for_ )
                || stored_quantization != priced_stored_quantization_ )
            {
                throw std::invalid_argument( std::format(
                    "{}: this plan was priced for {} '{}' stored as '{}', and '{}' holds {} '{}' stored as '{}'",
                    caller,
                    priced_for_.architecture, priced_for_.model_name, storedName( priced_stored_quantization_ ),
                    weights_path,
                    metadata.architecture, metadata.model_name, storedName( stored_quantization ) ) );
            }
        }

        static std::string_view nameOf( ContextLimit limit ) noexcept
        {
            switch ( limit )
            {
                case ContextLimit::FixedByCaller: return "FixedByCaller";
                case ContextLimit::TrainedMaximum: return "TrainedMaximum";
                case ContextLimit::FullPrefillChunk: return "FullPrefillChunk";
                case ContextLimit::DeviceMemory: return "DeviceMemory";
            }

            return "Unknown";
        }

        static std::string_view nameOf( PrefillChunkLimit limit ) noexcept
        {
            switch ( limit )
            {
                case PrefillChunkLimit::LargestTheContextPermits: return "LargestTheContextPermits";
                case PrefillChunkLimit::DeviceMemory: return "DeviceMemory";
            }

            return "Unknown";
        }

    private:

        static std::string pricingFacts( WeightsMetadata metadata )
        {
            metadata.model_name.clear();

            return toMetadataJSON( metadata );
        }

        static std::string_view storedName( std::string_view stored_quantization ) noexcept
        {
            return stored_quantization.empty() ? std::string_view{ "none" } : stored_quantization;
        }

        DeviceReading reading_;
        dim_t context_length_;
        ContextLimit context_limit_;
        PrefillChunking prefill_;
        MemoryStats footprint_;
        WeightQuantization weight_quantization_;
        KvCacheCompression kv_cache_compression_;
        dim_t language_model_head_positions_;
        WeightsMetadata priced_for_;
        std::string priced_stored_quantization_;
    };
}
