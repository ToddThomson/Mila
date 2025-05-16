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

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Dnn.TensorHelpers;
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
    export template<DeviceType TDeviceType = DeviceType::Cuda,
        typename TLogits = float,
        typename TTargets = int>
        requires ValidFloatTensorType<TLogits>
    class CrossEntropy : public Module<TDeviceType, TLogits, TTargets, TLogits> {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaMemoryResource, CpuMemoryResource>; ///< Memory resource type based on device type
        using ModuleBase = Module<TDeviceType, TLogits, TTargets, TLogits>; ///< Base class type for the module

        /**
         * @brief Construct a new CrossEntropy module with the default device context.
         *
         * @param name The name identifier for the module.
         * @param device_name The name of the device to use for this module.
         * @param vocab_size The size of the vocabulary (number of possible classes).
         * @param is_training Whether the module is in training mode. Default is false.
         */
        CrossEntropy( std::string name, const std::string& device_name, int64_t vocab_size, bool is_training = false )
            : ModuleBase( device_name ), vocab_size_( vocab_size ) {
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
         * @param is_training Whether the module is in training mode. Default is false.
         */
        CrossEntropy( std::string name, std::shared_ptr<DeviceContext> context, int64_t vocab_size, bool is_training = false )
            : ModuleBase( context ), vocab_size_( vocab_size ) {
            this->setTraining( is_training );
            this->setName( name );
            createOperation();
        }

        /**
         * @brief Get the number of parameters in the module.
         *
         * The CrossEntropy module has no trainable parameters.
         *
         * @return size_t Always returns 0 as CrossEntropy has no parameters.
         */
        size_t parameterCount() const override {
            return 0;
        }

        /**
         * @brief Perform the forward pass of the cross entropy operation.
         *
         * Computes the cross entropy loss between the predicted probabilities and the target classes.
         *
         * @param input The input tensor containing the predicted probabilities.
         * @param targets The tensor containing the target class indices.
         * @param output The tensor where the loss values will be stored.
         */
        void forward( const Tensor<TLogits, MR>& input, const Tensor<TTargets, MR>& targets, Tensor<TLogits, MR>& output ) {
            operation_->forward( targets, parameters_, attributes_, output, output_state_ );
        }

        /**
         * @brief Save the module's state to a zip archive.
         *
         * Serializes the module's state to the provided zip archive. Since CrossEntropy has no trainable parameters,
         * this primarily preserves the module's configuration.
         *
         * @param zip The zip archive where the module state will be saved.
         */
        void save( mz_zip_archive& zip ) const override {
            // CrossEntropy has no parameters to save
        }

        /**
         * @brief Load the module's state from a zip archive.
         *
         * Deserializes the module's state from the provided zip archive. Since CrossEntropy has no trainable parameters,
         * this primarily restores the module's configuration.
         *
         * @param zip The zip archive from which to load the module state.
         */
        void load( mz_zip_archive& zip ) override {
            // CrossEntropy has no parameters to load
        }

        /**
         * @brief Convert the module information to string.
         *
         * Creates a string representation of the CrossEntropy module, including its name,
         * vocabulary size, and the device.
         *
         * @return std::string A formatted string describing the module configuration.
         */
        std::string toString() const override {
            std::ostringstream oss;
            oss << "--------------------" << std::endl;
            oss << "CrossEntropy: " << this->getName() << ", Vocabulary Size: " << vocab_size_;
            oss << ", Device: " << deviceToString( this->getDeviceContext()->getDevice()->getDeviceType() ) << std::endl;

            return oss.str();
        }

    private:
        /**
         * @brief The size of the vocabulary (number of possible classes).
         */
        int64_t vocab_size_{ 0 };

        /**
         * @brief The parameters for the operation.
         *
         * The CrossEntropy operation has no parameters, so this is an empty vector.
         */
        std::vector<std::shared_ptr<Tensor<TLogits, MR>>> parameters_;

        /**
         * @brief The output state.
         *
         * Storage for intermediate results that might be needed for the backward pass.
         */
        std::vector<std::shared_ptr<Tensor<TLogits, MR>>> output_state_;

        /**
         * @brief The operation attributes.
         *
         * Contains the configuration for the cross entropy operation, such as the vocabulary size.
         */
        OperationAttributes attributes_;

        /**
         * @brief The underlying unary operation that implements the cross entropy function.
         */
        std::shared_ptr<UnaryOperation<TDeviceType, TLogits, TTargets, TLogits>> operation_{ nullptr };

        /**
         * @brief Create the appropriate cross entropy operation based on the current device context.
         *
         * This method initializes the operation_ member with the appropriate implementation
         * of the cross entropy operation for either CPU or CUDA, based on the current device context.
         */
        void createOperation() {
            // Set the vocabulary size in attributes
            attributes_.set( "vocab_size", vocab_size_ );

            if constexpr ( TDeviceType == DeviceType::Cpu ) {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cpu, TLogits, TTargets, TLogits>(
                    "Cpu::CrossEntropyOp",
                    this->getDeviceContext() );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cpu, TLogits, TTargets, TLogits>>(base_op);
            }
            else {
                auto base_op = OperationRegistry::instance().createUnaryOperation<DeviceType::Cuda, TLogits, TTargets, TLogits>(
                    "Cuda::CrossEntropyOp",
                    this->getDeviceContext() );

                operation_ = std::static_pointer_cast<UnaryOperation<DeviceType::Cuda, TLogits, TTargets, TLogits>>(base_op);
            }
        }
    };

    /**
     * @brief Type alias for CPU-based cross entropy module.
     *
     * @tparam TLogits Data type of the predicted probabilities (typically float).
     * @tparam TTargets Data type of the target class indices (typically int).
     */
    export template<typename TLogits = float, typename TTargets = int>
        using CpuCrossEntropy = CrossEntropy<DeviceType::Cpu, TLogits, TTargets>;

    /**
     * @brief Type alias for CUDA-based cross entropy module.
     *
     * @tparam TLogits Data type of the predicted probabilities (typically float).
     * @tparam TTargets Data type of the target class indices (typically int).
     */
    export template<typename TLogits = float, typename TTargets = int>
        using CudaCrossEntropy = CrossEntropy<DeviceType::Cuda, TLogits, TTargets>;
}