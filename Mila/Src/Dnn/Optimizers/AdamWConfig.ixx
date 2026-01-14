/**
 * @file AdamWConfig.ixx
 * @brief AdamW optimizer configuration.
 *
 * Fluent setters, validation and metadata serialization for AdamW hyperparameters.
 */

module;
#include <string>
#include <stdexcept>
#include <sstream>
#include <utility>

export module Dnn.Optimizers.AdamWConfig;

import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn::Optimizers
{
    using namespace Mila::Dnn;
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration for AdamW optimizer.
     *
     * Encapsulates hyperparameters for AdamW and provides fluent setters,
     * validation and conversion to/from the framework's metadata abstraction.
     */
    export class AdamWConfig : public ComponentConfig
    {
    public:
        // Fluent setters

        template <typename Self>
        decltype(auto) withLearningRate( this Self&& self, float learning_rate ) noexcept
        {
            self.learning_rate_ = learning_rate;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withBeta1( this Self&& self, float beta1 ) noexcept
        {
            self.beta1_ = beta1;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withBeta2( this Self&& self, float beta2 ) noexcept
        {
            self.beta2_ = beta2;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withEpsilon( this Self&& self, float epsilon ) noexcept
        {
            self.epsilon_ = epsilon;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withWeightDecay( this Self&& self, float weight_decay ) noexcept
        {
            self.weight_decay_ = weight_decay;
            return std::forward<Self>( self );
        }

        // Getters

        float getLearningRate() const noexcept
        {
            return learning_rate_;
        }

        float getBeta1() const noexcept
        {
            return beta1_;
        }

        float getBeta2() const noexcept
        {
            return beta2_;
        }

        float getEpsilon() const noexcept
        {
            return epsilon_;
        }

        float getWeightDecay() const noexcept
        {
            return weight_decay_;
        }

        // Validation

        /**
         * @brief Validate configuration parameters.
         *
         * Throws std::invalid_argument on invalid parameters.
         */
        void validate() const override
        {
            if ( learning_rate_ <= 0.0f )
            {
                throw std::invalid_argument( "AdamWConfig: learning rate must be positive" );
            }

            if ( beta1_ <= 0.0f || beta1_ >= 1.0f )
            {
                std::ostringstream oss;
                oss << "AdamWConfig: beta1 must be in (0, 1), got " << beta1_;
                throw std::invalid_argument( oss.str() );
            }

            if ( beta2_ <= 0.0f || beta2_ >= 1.0f )
            {
                std::ostringstream oss;
                oss << "AdamWConfig: beta2 must be in (0, 1), got " << beta2_;
                throw std::invalid_argument( oss.str() );
            }

            if ( epsilon_ <= 0.0f )
            {
                throw std::invalid_argument( "AdamWConfig: epsilon must be positive" );
            }

            if ( weight_decay_ < 0.0f )
            {
                throw std::invalid_argument( "AdamWConfig: weight decay must be non-negative" );
            }
        }

        // Metadata serialization (new API)

        /**
         * @brief Convert configuration into SerializationMetadata.
         *
         * Keys produced:
         * - "precision" : integer
         * - "learning_rate" : double
         * - "beta1" : double
         * - "beta2" : double
         * - "epsilon" : double
         * - "weight_decay" : double
         */
        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>( precision_ ) )
                .set( "learning_rate", learning_rate_ )
                .set( "beta1", beta1_ )
                .set( "beta2", beta2_ )
                .set( "epsilon", epsilon_ )
                .set( "weight_decay", weight_decay_ );

            return meta;
        }

        /**
         * @brief Populate configuration from provided metadata.
         *
         * Missing keys are ignored; type-safe try-get helpers are used so
         * incorrect/missing entries don't throw but simply leave defaults intact.
         */
        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto p = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_ )>( *p );
            }

            if ( auto v = meta.tryGetFloat( "learning_rate" ) )
            {
                learning_rate_ = *v;
            }

            if ( auto b1 = meta.tryGetFloat( "beta1" ) )
            {
                beta1_ = *b1;
            }

            if ( auto b2 = meta.tryGetFloat( "beta2" ) )
            {
                beta2_ = *b2;
            }

            if ( auto eps = meta.tryGetFloat( "epsilon" ) )
            {
                epsilon_ = *eps;
            }

            if ( auto wd = meta.tryGetFloat( "weight_decay" ) )
            {
                weight_decay_ = *wd;
            }
        }

        // String summary

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "AdamWConfig(learning_rate=" << learning_rate_
                << ", beta1=" << beta1_
                << ", beta2=" << beta2_
                << ", epsilon=" << epsilon_
                << ", weight_decay=" << weight_decay_ << "\")";

            return oss.str();
        }

    private:
        float learning_rate_{ 0.001f };
        float beta1_{ 0.9f };
        float beta2_{ 0.999f };
        float epsilon_{ 1e-8f };
        float weight_decay_{ 0.01f };
    };
}