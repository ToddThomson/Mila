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

export module Dnn.Optimizers.AdamW:Config;

import Dnn.ConfigurationBase;

namespace Mila::Dnn::Optimizers
{
    using namespace Mila::Dnn;

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
    export class AdamWConfig : public ConfigurationBase
    {
    public:
        /**
         * @brief Default constructor with standard AdamW hyperparameters.
         *
         * Initializes with commonly used default values from the literature:
         * - lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01
         */
        AdamWConfig()
            : learning_rate_( 0.001f )
            , beta1_( 0.9f )
            , beta2_( 0.999f )
            , epsilon_( 1e-8f )
            , weight_decay_( 0.01f )
        {
        }

        /**
         * @brief Construct with explicit learning rate.
         *
         * @param learning_rate Initial learning rate (must be positive)
         */
        explicit AdamWConfig( float learning_rate )
            : learning_rate_( learning_rate )
            , beta1_( 0.9f )
            , beta2_( 0.999f )
            , epsilon_( 1e-8f )
            , weight_decay_( 0.01f )
        {
        }

        /**
         * @brief Construct with full hyperparameter specification.
         *
         * @param learning_rate Initial learning rate
         * @param beta1 First moment decay rate
         * @param beta2 Second moment decay rate
         * @param epsilon Numerical stability constant
         * @param weight_decay Weight decay coefficient
         */
        AdamWConfig(
            float learning_rate,
            float beta1,
            float beta2,
            float epsilon,
            float weight_decay )
            : learning_rate_( learning_rate )
            , beta1_( beta1 )
            , beta2_( beta2 )
            , epsilon_( epsilon )
            , weight_decay_( weight_decay )
        {
        }

        // ====================================================================
        // Fluent Setters (Builder Pattern)
        // ====================================================================

        /**
         * @brief Set learning rate.
         *
         * @param learning_rate Learning rate (must be positive)
         * @return Reference to this config for method chaining
         */
        AdamWConfig& withLearningRate( float learning_rate ) noexcept
        {
            learning_rate_ = learning_rate;
            return *this;
        }

        /**
         * @brief Set beta1 (first moment decay rate).
         *
         * Controls exponential decay of first moment estimates (momentum).
         * Typical values: 0.9 (default), range (0, 1)
         *
         * @param beta1 First moment decay rate
         * @return Reference to this config for method chaining
         */
        AdamWConfig& withBeta1( float beta1 ) noexcept
        {
            beta1_ = beta1;
            return *this;
        }

        /**
         * @brief Set beta2 (second moment decay rate).
         *
         * Controls exponential decay of second moment estimates (variance).
         * Typical values: 0.999 (default), 0.99 for some tasks, range (0, 1)
         *
         * @param beta2 Second moment decay rate
         * @return Reference to this config for method chaining
         */
        AdamWConfig& withBeta2( float beta2 ) noexcept
        {
            beta2_ = beta2;
            return *this;
        }

        /**
         * @brief Set epsilon (numerical stability constant).
         *
         * Small constant added to denominator for numerical stability.
         * Typical values: 1e-8 (default), 1e-7 for some precisions
         *
         * @param epsilon Numerical stability constant
         * @return Reference to this config for method chaining
         */
        AdamWConfig& withEpsilon( float epsilon ) noexcept
        {
            epsilon_ = epsilon;
            return *this;
        }

        /**
         * @brief Set weight decay coefficient.
         *
         * Controls strength of L2 regularization in AdamW's decoupled weight decay.
         * Typical values: 0.01 (default), 0.0 to disable, 0.001-0.1 for tuning
         *
         * @param weight_decay Weight decay coefficient (non-negative)
         * @return Reference to this config for method chaining
         */
        AdamWConfig& withWeightDecay( float weight_decay ) noexcept
        {
            weight_decay_ = weight_decay;
            return *this;
        }

        /**
         * @brief Set optimizer name for identification.
         *
         * @param name Optimizer name
         * @return Reference to this config for method chaining
         */
        AdamWConfig& withName( const std::string& name ) noexcept
        {
            name_ = name;
            return *this;
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

        /**
         * @brief Get optimizer name.
         */
        const std::string& getName() const noexcept
        {
            return name_;
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

    private:
        float learning_rate_;
        float beta1_;
        float beta2_;
        float epsilon_;
        float weight_decay_;
        std::string name_{ "AdamW" };
    };
}