/**
 * @file CrossEntropyConfig.ixx
 * @brief Configuration interface for the CrossEntropy module in the Mila DNN framework.
 *
 * Defines the CrossEntropyConfig class, providing a type-safe fluent interface for configuring
 * CrossEntropy loss function modules. Inherits from ComponentConfig CRTP base and adds
 * CrossEntropy-specific options such as vocabulary size and weight handling.
 */

module;
#include <stdexcept>
#include <vector>
#include <cstdint>

export module Dnn.Modules.CrossEntropy:Config;

import Dnn.Module;
import Dnn.ComponentConfig;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for CrossEntropy module.
     */
    export class CrossEntropyConfig : public ComponentConfig<CrossEntropyConfig> {
    public:
        /**
         * @brief Constructor with required vocabulary size parameter.
         *
         * @param vocab_size The size of the vocabulary (number of possible classes)
         */
        explicit CrossEntropyConfig( int64_t vocab_size )
            : vocab_size_( vocab_size ) {}

        /**
         * @brief Configure whether to ignore padding index.
         *
         * When true, targets with the specified padding index will not contribute to the loss.
         *
         * @param ignore_pad Enable padding index ignoring
         * @return CrossEntropyConfig& Reference to this for method chaining
         */
        CrossEntropyConfig& withIgnorePadding( bool ignore_pad ) {
            ignore_padding_ = ignore_pad;
            return *this;
        }

        /**
         * @brief Set the padding index to ignore.
         *
         * @param pad_idx The padding index value to ignore in loss calculation
         * @return CrossEntropyConfig& Reference to this for method chaining
         */
        CrossEntropyConfig& withPaddingIndex( int64_t pad_idx ) {
            padding_idx_ = pad_idx;
            return *this;
        }

        /**
         * @brief Set class weights for weighted cross entropy.
         *
         * @param weights Vector of weights for each class
         * @return CrossEntropyConfig& Reference to this for method chaining
         */
        CrossEntropyConfig& withClassWeights( const std::vector<float>& weights ) {
            class_weights_ = weights;
            return *this;
        }

        /**
         * @brief Configure whether to reduce the loss.
         *
         * When true, returns the mean of losses. When false, returns per-sample losses.
         *
         * @param reduce Whether to average the loss
         * @return CrossEntropyConfig& Reference to this for method chaining
         */
        CrossEntropyConfig& withReduction( bool reduce ) {
            reduce_ = reduce;
            return *this;
        }

        /**
         * @brief Configure whether to apply label smoothing.
         *
         * @param smoothing Label smoothing factor (0.0 to 1.0)
         * @return CrossEntropyConfig& Reference to this for method chaining
         */
        CrossEntropyConfig& withLabelSmoothing( float smoothing ) {
            label_smoothing_ = smoothing;
            return *this;
        }

        /**
         * @brief Get the vocabulary size.
         */
        int64_t getVocabSize() const { return vocab_size_; }

        /**
         * @brief Check if padding should be ignored.
         */
        bool ignorePadding() const { return ignore_padding_; }

        /**
         * @brief Get the padding index.
         */
        int64_t getPaddingIndex() const { return padding_idx_; }

        /**
         * @brief Get the class weights.
         */
        const std::vector<float>& getClassWeights() const { return class_weights_; }

        /**
         * @brief Check if loss should be reduced.
         */
        bool getReduction() const { return reduce_; }

        /**
         * @brief Get the label smoothing factor.
         */
        float getLabelSmoothing() const { return label_smoothing_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
         */
        void validate() const {
            ComponentConfig<CrossEntropyConfig>::validate();

            if ( vocab_size_ <= 0 ) {
                throw std::invalid_argument( "Vocabulary size must be greater than zero" );
            }

            if ( ignore_padding_ && (padding_idx_ < 0 || padding_idx_ >= vocab_size_) ) {
                throw std::invalid_argument( "Padding index must be within valid vocabulary range" );
            }

            if ( !class_weights_.empty() && class_weights_.size() != static_cast<size_t>( vocab_size_ ) ) {
                throw std::invalid_argument( "Class weights size must match vocabulary size" );
            }

            if ( label_smoothing_ < 0.0f || label_smoothing_ > 1.0f ) {
                throw std::invalid_argument( "Label smoothing factor must be between 0 and 1" );
            }
        }

    private:
        int64_t vocab_size_;
        bool ignore_padding_ = false;
        int64_t padding_idx_ = -1;
        std::vector<float> class_weights_;
        bool reduce_ = true;
        float label_smoothing_ = 0.0f;
    };
}