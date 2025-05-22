/**
 * @file CrossEntropy.ixx
 * @brief Implementation of the CrossEntropy loss function module.
 */

module;
#include <miniz.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <cstdint>

export module Dnn.Modules.CrossEntropy;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;
import Compute.Precision;
import Compute.DeviceType;
import Compute.DeviceContext;
import Compute.ComputeDevice;
import Compute.CpuDevice;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationAttributes;
import Compute.OperationRegistry;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaMemoryResource;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief CrossEntropy loss module for neural networks.
     *
     * This class implements the cross entropy loss function, which is commonly used in classification tasks.
     * It computes the negative log likelihood of the correct class given the predicted probabilities.
     *
     * The cross entropy loss for a single example is calculated as:
     * -log(p_i) where p_i is the predicted probability for the correct class i.
     *
     * @tparam TDeviceType The device type (CPU or CUDA) on which to perform computations.
     * @tparam TLogits The data type of the predicted probabilities (typically float).
     * @tparam TTargets The data type of the target class indices (typically int).
     */
    export template<DeviceType TDeviceType = DeviceType::Cuda, typename TLogits = float, typename TTargets = int>
        requires ValidFloatTensorType<TLogits>
    class CrossEntropy : public Module<TDeviceType, TLogits, TTargets> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>;
        using ModuleBase = Module<TDeviceType, TLogits, TTargets>;

        /**
         * @brief Construct a new CrossEntropy module from configuration.
         *
         * @param config The configuration for this module
         */
        explicit CrossEntropy( const CrossEntropyConfig& config )
            : ModuleBase(
                config.getContext() ? config.getContext() : std::make_shared<DeviceContext>( config.getDeviceName() ),
                TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : config.getPrecision() ),
            vocab_size_( config.getVocabSize() ),
            ignore_padding_( config.ignorePadding() ),
            padding_idx_( config.getPaddingIndex() ),
            reduce_( config.getReduction() ),
            label_smoothing_( config.getLabelSmoothing() ) {

            config.validate();

            this->setName( config.getName() );
            this->setTraining( config.isTraining() );

            const auto& weights = config.getClassWeights();
            if ( !weights.empty() ) {
                initializeClassWeights( weights );
            }

            createOperation();
        }

        /**
         * @brief Construct a new CrossEntropy module with the default device context.
         *
         * @param name The name identifier for the module.
         * @param device_name The name of the device to use for this module.
         * @param vocab_size The size of the vocabulary (number of possible classes).
         * @param precision The compute precision policy (CPU operations always use Disabled).
         * @param is_training Whether the module is in training mode. Default is false.
         */
        CrossEntropy( std::string name, const std::string& device_name, int64_t vocab_size,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto,
            bool is_training = false )
            : ModuleBase( device_name, TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : precision ),
            vocab_size_( vocab_size ) {
            this->setTraining( is_training );
            this->setName( name );
            createOperation();
        }

        /**
         * @brief Construct a new CrossEntropy module with a specific device context.
         *
         * @param name The name identifier for the module.
         * @param context The device context to use for this module.
         * @param vocab_size The size of the vocabulary (number of possible classes).
         * @param precision The compute precision policy (CPU operations always use Disabled).
         * @param is_training Whether the module is in training mode. Default is false.
         */
        CrossEntropy( std::string name, std::shared_ptr<DeviceContext> context, int64_t vocab_size,
            ComputePrecision::Policy precision = ComputePrecision::Policy::Auto,
            bool is_training = false )
            : ModuleBase( context, TDeviceType == DeviceType::Cpu ? ComputePrecision::Policy::Disabled : precision ),
            vocab_size_( vocab_size ) {
            this->setTraining( is_training );
            this->setName( name );
            createOperation();
        }

        /**
         * @brief Get the number of parameters in the module.
         */
        size_t parameterCount() const override {
            return class_weights_ ? class_weights_->size() : 0;
        }

        /**
         * @brief Perform the forward pass of the cross entropy operation.
         */
        void forward( const Tensor<TLogits, MR>& input, const Tensor<TTargets, MR>& targets, Tensor<TLogits, MR>& output ) {
            operation_->forward( input, targets, parameters_, attributes_, output, output_state_ );
        }

        /**
         * @brief Calculate gradients for the backward pass.
         */
        void backward(
            const Tensor<TLogits, MR>& input,
            const Tensor<TTargets, MR>& targets,
            const Tensor<TLogits, MR>& output_grad,
            Tensor<TLogits, MR>& input_grad ) {

            operation_->backward(
                input,           // Input tensor (logits)
                targets,         // Target classes
                parameters_,     // Optional class weights
                {},              // No parameter gradients needed
                output_grad,     // Gradient from next layer
                input_grad,      // Gradient to propagate to logits
                attributes_,     // Operation attributes
                output_state_    // Cached tensors from forward pass
            );
        }

        /**
         * @brief Get the vocabulary size.
         */
        int64_t getVocabSize() const { return vocab_size_; }

        /**
         * @brief Check if padding is ignored.
         */
        bool ignorePadding() const { return ignore_padding_; }

        /**
         * @brief Get the padding index.
         */
        int64_t getPaddingIndex() const { return padding_idx_; }

        /**
         * @brief Check if loss is reduced.
         */
        bool isReduced() const { return reduce_; }

        /**
         * @brief Get the label smoothing factor.
         */
        float getLabelSmoothing() const { return label_smoothing_; }

        /**
         * @brief Get the class weights tensor.
         */
        std::shared_ptr<Tensor<TLogits, MR>> getClassWeights() const {
            return class_weights_;
        }

        /**
         * @brief Save the module's state to a zip archive.
         */
        void save( mz_zip_archive& zip ) const override {
            if ( class_weights_ ) {
                // Save class weights if present
                // Implementation depends on tensor serialization
            }
        }

        /**
         * @brief Load the module's state from a zip archive.
         */
        void load( mz_zip_archive& zip ) override {
            if ( class_weights_ ) {
                // Load class weights if present
                // Implementation depends on tensor deserialization
            }
        }

        /**
         * @brief Convert the module information to string.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "CrossEntropy: " << this->getName() << ", Vocabulary Size: " << vocab_size_ << std::endl;

            if ( ignore_padding_ ) {
                oss << "Ignoring padding at index: " << padding_idx_ << std::endl;
            }

            if ( class_weights_ ) {
                oss << "Using class weights" << std::endl;
            }

            oss << "Reduction: " << (reduce_ ? "Mean" : "None") << std::endl;

            if ( label_smoothing_ > 0.0f ) {
                oss << "Label smoothing: " << label_smoothing_ << std::endl;
            }

            oss << "Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;
            oss << this->getComputePrecision().toString() << std::endl;

            return oss.str();
        }

    private:
        int64_t vocab_size_{ 0 };
        bool ignore_padding_{ false };
        int64_t padding_idx_{ -1 };
        bool reduce_{ true };
        float label_smoothing_{ 0.0f };

        std::shared_ptr<Tensor<TLogits, MR>> class_weights_{ nullptr };
        std::vector<std::shared_ptr<Tensor<TLogits, MR>>> parameters_;
        std::vector<std::shared_ptr<Tensor<TLogits, MR>>> output_state_;
        OperationAttributes attributes_;
        std::shared_ptr<UnaryOperation<TDeviceType, TLogits, TTargets>> operation_{ nullptr };

        void initializeClassWeights( const std::vector<float>& weights ) {
            class_weights_ = std::make_shared<Tensor<TLogits, MR>>( std::vector<size_t>{static_cast<size_t>(vocab_size_)} );
            class_weights_->setName( this->getName() + ".class_weights" );

            // Copy the weights into the tensor
            // This is a simplified placeholder - actual implementation would copy the data to device

            parameters_.clear();
            parameters_.push_back( class_weights_ );
        }

        void createOperation() {
            attributes_.set( "vocab_size", vocab_size_ );

            if ( ignore_padding_ ) {
                attributes_.set( "ignore_padding", true );
                attributes_.set( "padding_idx", padding_idx_ );
            }

            attributes_.set( "reduce", reduce_ );

            if ( label_smoothing_ > 0.0f ) {
                attributes_.set( "label_smoothing", label_smoothing_ );
            }

            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TLogits, TTargets>(
                    "Cpu::CrossEntropyOp",
                    this->getDeviceContext(),
                    ComputePrecision::Policy::Disabled );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TLogits, TTargets>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TLogits, TTargets>(
                    "Cuda::CrossEntropyOp",
                    this->getDeviceContext(),
                    this->getComputePrecision().getPolicy() );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TLogits, TTargets>>(base_op);
            }
        }
    };

    export template<typename TLogits = float, typename TTargets = int>
        using CpuCrossEntropy = CrossEntropy<DeviceType::Cpu, TLogits, TTargets>;

    export template<typename TLogits = float, typename TTargets = int>
        using CudaCrossEntropy = CrossEntropy<DeviceType::Cuda, TLogits, TTargets>;
}