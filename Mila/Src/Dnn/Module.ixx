module;
#include <vector>
#include <string>
#include <memory>

export module Dnn.Module;

import Dnn.Tensor;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.DeviceMemoryResource;

namespace Mila::Dnn
{
    /**
     * @brief Abstract base class for all modules in the DNN framework.
     * 
     * @tparam T Data type of the tensor elements.
     * @tparam MR Memory resource type, either CpuMemoryResource or DeviceMemoryResource.
     */
    export 
    template<typename T, typename MR> requires std::is_same_v<MR, Compute::CpuMemoryResource> || std::is_same_v<MR, Compute::DeviceMemoryResource>
    class Module {
    public:
        /// Virtual destructor.
        virtual ~Module() = default;

        /**
         * @brief Forward pass of the module.
         * 
         * @param input Input tensor.
         * @return std::shared_ptr<Tensor<T, MR>> Output tensor.
         */
        virtual std::shared_ptr<Tensor<T, MR>> forward( const std::shared_ptr<Tensor<T,MR>> input ) = 0;

        /**
         * @brief Backward pass of the module.
         * 
         * @param gradient Gradient tensor.
         * @return Tensor<T, MR> Gradient with respect to the input.
         */
        virtual Tensor<T,MR> backward( const Tensor<T,MR>& gradient ) {
            // Default to no op
            return {};
        }

        /**
         * @brief Set the training mode of the module.
         * 
         * @param training True if the module is in training mode, false otherwise.
         */
        void setTrainingMode( bool training ) {
            is_training_ = training;
        }

        /**
         * @brief Check if the module is in training mode.
         * 
         * @return true If the module is in training mode.
         * @return false Otherwise.
         */
        bool isTraining() const {
            return is_training_;
        }

        /**
         * @brief Get the number of parameters in the module.
         * 
         * @return size_t Number of parameters.
         */
        virtual size_t parameters() const = 0;

        /**
         * @brief Get the name of the module.
         * 
         * @return std::string Name of the module.
         */
        virtual std::string name() const = 0;

        /**
         * @brief Print the module information.
         */
        virtual void print() const = 0;

    private:
        bool is_training_{ false }; ///< Training mode flag.
    };
}