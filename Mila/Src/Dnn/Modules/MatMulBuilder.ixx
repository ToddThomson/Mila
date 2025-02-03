module;
#include <memory>
#include <string>

export module Dnn.Modules.MatMulBuilder;

import Dnn.Modules.MatMul;

export namespace Mila::Dnn::Modules
{
    /**
     * @brief A builder class for the MatMul module.
     *
     * @tparam T The data type of the module.
     */
    export template<typename T>
        class MatMulBuilder {
        public:
            /**
             * @brief Set the name of the module.
             *
             * @param name The name of the module.
             * @return MatMulBuilder& The builder instance.
             */
            MatMulBuilder& setName( const std::string& name ) {
                name_ = name;
                return *this;
            }

            /**
             * @brief Set the batch size.
             *
             * @param batch_size The batch size.
             * @return MatMulBuilder& The builder instance.
             */
            MatMulBuilder& setBatchSize( int64_t batch_size ) {
                batch_size_ = batch_size;
                return *this;
            }

            /**
             * @brief Set the sequence length.
             *
             * @param sequence_length The sequence length.
             * @return MatMulBuilder& The builder instance.
             */
            MatMulBuilder& setSequenceLength( int64_t sequence_length ) {
                sequence_length_ = sequence_length;
                return *this;
            }

            /**
             * @brief Set the number of channels.
             *
             * @param channels The number of channels.
             * @return MatMulBuilder& The builder instance.
             */
            MatMulBuilder& setChannels( int64_t channels ) {
                channels_ = channels;
                return *this;
            }

            /**
             * @brief Set the number of output channels.
             *
             * @param output_channels The number of output channels.
             * @return MatMulBuilder& The builder instance.
             */
            MatMulBuilder& setOutputChannels( int64_t output_channels ) {
                output_channels_ = output_channels;
                return *this;
            }

            /**
             * @brief Set the training mode.
             *
             * @param is_training Whether the module is in training mode.
             * @return MatMulBuilder& The builder instance.
             */
            MatMulBuilder& setIsTraining( bool is_training ) {
                is_training_ = is_training;
                return *this;
            }

            /**
             * @brief Build the MatMul module.
             *
             * @return std::shared_ptr<MatMul<T>> The built MatMul module.
             */
            std::shared_ptr<MatMul<T>> build() const {
                return std::make_shared<MatMul<T>>( name_, batch_size_, sequence_length_, channels_, output_channels_, is_training_ );
            }

        private:
            std::string name_{ "MatMul" }; ///< The name of the module.
            int64_t batch_size_{ 0 }; ///< The batch size.
            int64_t sequence_length_{ 0 }; ///< The sequence length.
            int64_t channels_{ 0 }; ///< The number of channels.
            int64_t output_channels_{ 0 }; ///< The number of output channels.
            bool is_training_{ false }; ///< Whether the module is in training mode.
    };
}
