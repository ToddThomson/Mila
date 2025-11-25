/**
 * @file AdamWConfig.ixx
 * @brief Configuration for AdamW optimizer.
 *
 * Provides fluent configuration API for AdamW optimizer hyperparameters.
 */

module;
#include <string>
#include <stdexcept>
#include <sstream>
#include <utility>

export module Dnn.Optimizers.AdamWConfig;

import Dnn.ComponentConfig;
import nlohmann.json;

namespace Mila::Dnn::Optimizers
{
    using namespace Mila::Dnn;
    using json = nlohmann::json;

    /**
     * @brief Configuration for AdamW optimizer.
     *
     * Encapsulates all hyperparameters for the AdamW optimization algorithm.
     * Provides fluent setters following the builder pattern for easy configuration.
     *
     * Default values are based on the original Adam/AdamW papers and common practice:
     * - Learning rate: 0.001 (1e-3)
     * - Beta1: 0.9 (first moment decay)
     * - Beta2: 0.999 (second moment decay)
     * - Epsilon: 1e-8 (numerical stability)
     * - Weight decay: 0.01 (L2 regularization strength)
     *
     * Example usage:
     * @code
     * auto config = AdamWConfig()
     *     .withLearningRate(0.001f)
     *     .withBeta1(0.9f)
     *     .withBeta2(0.999f)
     *     .withWeightDecay(0.01f);
     *
     * auto optimizer = std::make_shared<AdamWOptimizer<DeviceType::Cuda, TensorDataType::FP32>>(
     *     exec_context, config);
     * @endcode
     */
    export class AdamWConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Default constructor.
         */
        AdamWConfig()
        {
			this->name_ = "AdamW";
        }

        // ====================================================================
        // Fluent Setters
        // ====================================================================

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

        // ====================================================================
        // Getters
        // ====================================================================

        /**
         * @brief Get learning rate.
         */
        float getLearningRate() const noexcept
        {
            return learning_rate_;
        }

        /**
         * @brief Get beta1 parameter.
         */
        float getBeta1() const noexcept
        {
            return beta1_;
        }

        /**
         * @brief Get beta2 parameter.
         */
        float getBeta2() const noexcept
        {
            return beta2_;
        }

        /**
         * @brief Get epsilon parameter.
         */
        float getEpsilon() const noexcept
        {
            return epsilon_;
        }

        /**
         * @brief Get weight decay coefficient.
         */
        float getWeightDecay() const noexcept
        {
            return weight_decay_;
        }

        // ====================================================================
        // Validation
        // ====================================================================

        /**
         * @brief Validate configuration parameters.
         *
         * Ensures all hyperparameters are within valid ranges:
         * - learning_rate > 0
         * - 0 < beta1 < 1
         * - 0 < beta2 < 1
         * - epsilon > 0
         * - weight_decay >= 0
         *
         * @throws std::invalid_argument if any parameter is invalid
         */
        void validate() const override
        {
            ComponentConfig::validate();

            if (learning_rate_ <= 0.0f)
            {
                throw std::invalid_argument( "AdamWConfig: learning rate must be positive" );
            }

            if (beta1_ <= 0.0f || beta1_ >= 1.0f)
            {
                std::ostringstream oss;
                oss << "AdamWConfig: beta1 must be in (0, 1), got " << beta1_;
                throw std::invalid_argument( oss.str() );
            }

            if (beta2_ <= 0.0f || beta2_ >= 1.0f)
            {
                std::ostringstream oss;
                oss << "AdamWConfig: beta2 must be in (0, 1), got " << beta2_;
                throw std::invalid_argument( oss.str() );
            }

            if (epsilon_ <= 0.0f)
            {
                throw std::invalid_argument( "AdamWConfig: epsilon must be positive" );
            }

            if (weight_decay_ < 0.0f)
            {
                throw std::invalid_argument( "AdamWConfig: weight decay must be non-negative" );
            }
        }

        // ====================================================================
        // JSON Serialization (ModuleConfig interface)
        // ====================================================================

        /**
         * @brief Serialize configuration to JSON.
         *
         * Keys:
         * - "name" : string
         * - "precision" : integer (underlying value of ComputePrecision::Policy)
         * - "learning_rate" : float
         * - "beta1" : float
         * - "beta2" : float
         * - "epsilon" : float
         * - "weight_decay" : float
         */
        json toJson() const
        {
            json j;
            j["name"] = name_;
            j["precision"] = static_cast<int>( precision_ );
            j["learning_rate"] = learning_rate_;
            j["beta1"] = beta1_;
            j["beta2"] = beta2_;
            j["epsilon"] = epsilon_;
            j["weight_decay"] = weight_decay_;

            return j;
        }

        /**
         * @brief Deserialize configuration from JSON.
         *
         * Missing keys leave fields at their current values. Type errors are
         * propagated from nlohmann::json getters.
         */
        void fromJson( const json& j )
        {
            if ( j.contains( "name" ) )
            {
                name_ = j.at( "name" ).get<std::string>();
            }

            if ( j.contains( "precision" ) )
            {
                precision_ = static_cast<decltype(precision_)>( j.at( "precision" ).get<int>() );
            }

            if ( j.contains( "learning_rate" ) )
            {
                learning_rate_ = j.at( "learning_rate" ).get<float>();
            }

            if ( j.contains( "beta1" ) )
            {
                beta1_ = j.at( "beta1" ).get<float>();
            }

            if ( j.contains( "beta2" ) )
            {
                beta2_ = j.at( "beta2" ).get<float>();
            }

            if ( j.contains( "epsilon" ) )
            {
                epsilon_ = j.at( "epsilon" ).get<float>();
            }

            if ( j.contains( "weight_decay" ) )
            {
                weight_decay_ = j.at( "weight_decay" ).get<float>();
            }
        }

        /**
         * @brief Produce a short, human-readable summary of this configuration.
         *
         * Overrides ComponentConfig::toString() to include AdamW-specific fields.
		 */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "AdamWConfig(learning_rate=" << learning_rate_
                << ", beta1=" << beta1_
                << ", beta2=" << beta2_
                << ", epsilon=" << epsilon_
                << ", weight_decay=" << weight_decay_
                << ", name=\"" << name_ << "\")";
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