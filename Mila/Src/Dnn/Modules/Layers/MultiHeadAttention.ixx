module;
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>

export module Dnn.Modules.Attention;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;

import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.CpuDevice;
import Compute.CudaDevice;

import Compute.OperationBase;
import Compute.OperationAttributes;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    export
        template<typename TInput, typename TPrecision = TInput>
        requires ValidTensorTypes<TInput, TPrecision>
    class MultiHeadAttention : public Module<TInput, TPrecision> {
    public:
        /**
         * @brief Constructs a new MultiHeadAttention module with the default device context.
         *
         * @param name The name of the module for identification purposes.
         * @param input_shape The shape of the input tensor.
         * @param num_heads The number of attention heads.
         * @param is_training Whether the module is initially in training mode. Default is false.
         */
        MultiHeadAttention( std::string name, const std::vector<size_t>& input_shape, size_t num_heads, bool is_training = false )
            : Module<TInput, TPrecision>(),
            input_shape_{ input_shape },
            num_heads_{ num_heads } {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
        }

        /**
         * @brief Constructs a new MultiHeadAttention module with a specific device context.
         *
         * @param name The name of the module for identification purposes.
         * @param input_shape The shape of the input tensor.
         * @param num_heads The number of attention heads.
         * @param context The device context to use for this module.
         * @param is_training Whether the module is initially in training mode. Default is false.
         */
        MultiHeadAttention( std::string name, const std::vector<size_t>& input_shape, size_t num_heads,
            std::shared_ptr<DeviceContext> context, bool is_training = false )
            : Module<TInput, TPrecision>( context ),
            input_shape_{ input_shape },
            num_heads_{ num_heads } {
            this->setTraining( is_training );
            this->setName( name );
            initializeTensors();
            createOperation();
        }

        /**
        * @brief Get the number of parameters.
        *
        * @return size_t The number of parameters.
        */
        size_t parameterCount() const override {
            return 0;
        }

        /**
         * @brief Performs the forward pass of the MultiHeadAttention operation.
         *
         * @param input The input tensor to be processed.
         * @param output The output tensor where the results will be stored.
         */
        template<typename TMR>
        void forward( const Tensor<TInput, TMR>& input, Tensor<TPrecision, TMR>& output ) {
            operation_->forward( input, parameters_, properties_, output, output_state_ );
        }

        /**
         * @brief Saves the module state to a ZIP archive.
         *
         * @param zip The ZIP archive to save the module state to.
         */
        void save( mz_zip_archive& zip ) const override {
            // MultiHeadAttention has no parameters to save
        }

        /**
         * @brief Loads the module state from a ZIP archive.
         *
         * @param zip The ZIP archive to load the module state from.
         */
        void load( mz_zip_archive& zip ) override {
            // MultiHeadAttention has no parameters to load
        }

        /**
        * @brief Convert the module information to string.
        *
        * @return std::string Module information as string.
        */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "MultiHeadAttention: " << this->getName();
            oss << ", Number of heads: " << num_heads_;
            oss << ", Input shape: (";
            for ( size_t i = 0; i < input_shape_.size(); ++i ) {
                oss << input_shape_[ i ];
                if ( i != input_shape_.size() - 1 ) {
                    oss << ",";
                }
            }
            oss << ")";
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->stateToString();

            return oss.str();
        }

    protected:
        /**
         * @brief Called when the device context changes.
         *
         * Recreates tensors and operations for the new device.
         */
        void onDeviceChanged() override {
            initializeTensors();
            createOperation();
        }

    private:
        std::vector<size_t> input_shape_; ///< The input shape.
        size_t num_heads_{ 0 }; ///< The number of attention heads.

        /**
         * @brief Tensor storing the attention weights after softmax.
         */
        std::shared_ptr<Tensor<TPrecision, typename Module<TInput, TPrecision>::MR>> attn_{ nullptr };

        /**
         * @brief Tensor storing the pre-softmax attention values.
         */
        std::shared_ptr<Tensor<TPrecision, typename Module<TInput, TPrecision>::MR>> pre_attn_{ nullptr };

        std::vector<std::shared_ptr<Tensor<TPrecision, typename Module<TInput, TPrecision>::MR>>> parameters_; ///< The parameters.
        std::vector<std::shared_ptr<Tensor<TPrecision, typename Module<TInput, TPrecision>::MR>>> output_state_; ///< The output state.
        std::vector<std::shared_ptr<Tensor<TPrecision, typename Module<TInput, TPrecision>::MR>>> scalars_; ///< The scalars.
        OperationAttributes properties_; ///< The operation properties.

        /**
         * @brief The underlying operation that implements the multi-head attention.
         */
        std::shared_ptr<UnaryOperation<TInput, TPrecision>> operation_{ nullptr };

        /**
         * @brief Initializes the tensors needed for the MultiHeadAttention operation.
         */
        void initializeTensors() {
            // Clear existing state
            output_state_.clear();

            auto batch_size = input_shape_[ 0 ];
            auto sequence_length = input_shape_[ 1 ];

            // Get device type for proper tensor creation
            auto device_type = this->getDeviceContext()->getDevice()->getDeviceType();

            // preatt, att are (B, NH, TDataType, TDataType). NH = number of heads, TDataType = sequence length
            if ( device_type == DeviceType::Cpu ) {
                pre_attn_ = std::make_shared<Tensor<TPrecision, Compute::HostMemoryResource>>(
                    std::vector<size_t>{batch_size, num_heads_, sequence_length, sequence_length} );
                pre_attn_->setName( this->getName() + ".pre_attn" );

                attn_ = std::make_shared<Tensor<TPrecision, Compute::HostMemoryResource>>(
                    std::vector<size_t>{batch_size, num_heads_, sequence_length, sequence_length} );
                attn_->setName( this->getName() + ".attn" );
            }
            else {
                pre_attn_ = std::make_shared<Tensor<TPrecision, Compute::DeviceMemoryResource>>(
                    std::vector<size_t>{batch_size, num_heads_, sequence_length, sequence_length} );
                pre_attn_->setName( this->getName() + ".pre_attn" );

                attn_ = std::make_shared<Tensor<TPrecision, Compute::DeviceMemoryResource>>(
                    std::vector<size_t>{batch_size, num_heads_, sequence_length, sequence_length} );
                attn_->setName( this->getName() + ".attn" );
            }

            // Add state tensors
            output_state_.emplace_back( pre_attn_ );
            output_state_.emplace_back( attn_ );

            // Set number of heads in properties
            properties_.setInt64( "num_heads", static_cast<int64_t>(num_heads_) );
        }

        /**
         * @brief Creates the appropriate MultiHeadAttention operation based on the current device context.
         */
        void createOperation() {
            // Get the device type from the context
            auto device_type = this->getDeviceContext()->getDevice()->getDeviceType();

            if ( operation_ ) {
                operation_.reset(); // Clear existing operation
            }

            if ( device_type == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cpu>( "Cpu::MultiHeadAttentionOp" );
                operation_ = std::static_pointer_cast<UnaryOperation<TInput, TPrecision>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createOperation<TInput, TPrecision, DeviceType::Cuda>( "Cuda::MultiHeadAttentionOp" );
                operation_ = std::static_pointer_cast<UnaryOperation<TInput, TPrecision>>(base_op);
            }
        }
    };
}

