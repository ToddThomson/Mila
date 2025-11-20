/**
 * @file Convolution.ixx
 * @brief Device-templated 2D convolution module.
 *
 * Thin module that owns convolution weights/bias and delegates compute to a
 * device-specific backend operation (registered in OperationRegistry, expected
 * name "Conv2DOp"). Supports training/eval transitions and parameter gradient
 * allocation consistent with other modules (LayerNorm, Linear).
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

export module Dnn.Modules.Layers.Convolution;
export import :Config;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.Precision;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.BinaryOperation;
import Compute.OperationRegistry;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Simple 2D convolution module.
     *
     * Notes:
     * - Relies on a backend operation named "Conv2DOp" registered for the
     *   device/precision. The backend is expected to expose forward and backward
     *   semantics compatible with this module.
     * - Parameters: `weight` shaped [out_channels, in_channels/groups, kh, kw]
     *   and optional `bias` shaped [out_channels].
     * - Gradient buffers are allocated when entering training mode and freed on
     *   leaving training mode (to save memory during inference).
     *
     * This class follows the Module::moduleSetTrainingImpl hook pattern: subclasses
     * implement `onEnterTraining()` / `onExitTraining()` to manage gradient buffers
     * and backend binding.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Convolution : public Module<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        using ConvOpType = BinaryOperation<TDeviceType, TPrecision>;

        /**
         * @param exec_context Execution context (must be non-null)
         * @param config       ConvolutionConfig describing kernel, stride, padding, groups, bias, etc.
         *
         * ConvolutionConfig is expected to be part of the repository's Config
         * types. The module will query OperationRegistry to create a backend
         * "Conv2DOp" operation using the same config object.
         */
        explicit Convolution( std::shared_ptr<ExecutionContextType> exec_context, const ConvolutionConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            createOperation();
        }

        ~Convolution() override = default;

        bool isBuilt() const override
        {
            return (operation_ != nullptr) && weight_ && (!config_.withBias() || bias_) && built_;
        }

        void build( const shape_t& input_shape ) override
        {
            if (built_)
            {
                throw std::runtime_error( "Convolution::build: module already built" );
            }

            // Validate input rank (expect NHWC or NCHW depending on config; delegate to config)
            config_.validateInputShape( input_shape );

            // Determine weight shape from config and input (helper in config assumed)
            shape_t wshape = config_.computeWeightShape( input_shape );

            auto device = exec_context_->getDevice();
            weight_ = std::make_shared<TensorType>( device, wshape );
            weight_->setName( this->getName() + ".weight" );

            if (config_.withBias())
            {
                shape_t bshape = { static_cast<int64_t>(config_.getOutChannels()) };
                bias_ = std::make_shared<TensorType>( device, bshape );
                bias_->setName( this->getName() + ".bias" );
            }

            // Bind parameters to backend
            operation_->setParameters( weight_.get(), bias_.get() );

            // If currently in training mode, ensure grads exist and bind them
            if (this->isTraining())
            {
                initializeParameterGradients();
                operation_->setParameterGradients( weight_grad_.get(), bias_grad_.get() );
            }

            operation_->build( input_shape );

            built_ = true;
        }

        // Forward semantics: input -> output (backend does convolution using bound params)
        void forward( const ITensor& input, ITensor& output )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Convolution must be built before forward()" );
            }

            operation_->forward( input, output );
        }

        // Backward: propagate through conv; requires training mode and gradients bound in backend
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "Convolution must be built before backward()" );
            }

            if (!this->isTraining())
            {
                throw std::runtime_error( "Convolution must be in training mode to call backward()" );
            }

            operation_->backward( input, output_grad, input_grad );
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> p;
            if (weight_) p.push_back( weight_.get() );
            if (bias_) p.push_back( bias_.get() );
            return p;
        }

        std::vector<ITensor*> getGradients() const override
        {
            if (!this->isTraining())
            {
                throw std::runtime_error( "Convolution: getGradients called when not in training mode" );
            }

            std::vector<ITensor*> g;
            if (weight_grad_) g.push_back( weight_grad_.get() );
            if (bias_grad_) g.push_back( bias_grad_.get() );
            return g;
        }

        size_t parameterCount() const override
        {
            size_t cnt = 0;
            if (weight_) cnt += weight_->size();
            if (bias_) cnt += bias_->size();
            return cnt;
        }

        std::string getName() const override
        {
            return config_.getName();
        }

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return exec_context_->getDevice();
        }

        void synchronize() override
        {
            exec_context_->synchronize();
        }

        void save( ModelArchive& archive ) const override
        {
            (void)archive;
        }

        void load( ModelArchive& archive ) override
        {
            (void)archive;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Convolution: " << getName() << " OC=" << config_.getOutChannels()
                << " KC=" << config_.getKernelHeight() << "x" << config_.getKernelWidth();
            return oss.str();
        }

    protected:
        // Called by Module::moduleSetTrainingImpl when entering training mode.
        void onEnterTraining() override
        {
            // inform backend op
            operation_->setTraining( true );

            if (built_)
            {
                initializeParameterGradients();
                operation_->setParameterGradients( weight_grad_.get(), config_.withBias() ? bias_grad_.get() : nullptr );
            }
        }

        // Called by Module::moduleSetTrainingImpl when leaving training mode.
        void onExitTraining() override
        {
            operation_->setTraining( false );
            operation_->setParameterGradients( nullptr, nullptr );

            // free gradient buffers to save memory
            weight_grad_.reset();
            bias_grad_.reset();
        }

    private:
        ConvolutionConfig config_;
        bool built_{ false };

        std::shared_ptr<TensorType> weight_{ nullptr };
        std::shared_ptr<TensorType> bias_{ nullptr };

        // Gradient buffers (allocated only in training)
        std::shared_ptr<TensorType> weight_grad_{ nullptr };
        std::shared_ptr<TensorType> bias_grad_{ nullptr };

        std::shared_ptr<ConvOpType> operation_{ nullptr };
        std::shared_ptr<ExecutionContextType> exec_context_;

        void createOperation()
        {
            operation_ = OperationRegistry::instance()
                .createBinaryOperation<TDeviceType, TPrecision>(
                    "Conv2DOp",
                    exec_context_,
                    config_ );

            if (!operation_)
            {
                throw std::runtime_error( "Convolution: failed to create backend Conv2DOp" );
            }
        }

        void initializeParameterGradients()
        {
            auto device = exec_context_->getDevice();

            if (!weight_grad_ && weight_)
            {
                weight_grad_ = std::make_shared<TensorType>( device, weight_->shape() );
                weight_grad_->setName( this->getName() + ".weight.grad" );
                zeros( *weight_grad_ );
            }

            if (config_.withBias() && !bias_grad_ && bias_)
            {
                bias_grad_ = std::make_shared<TensorType>( device, bias_->shape() );
                bias_grad_->setName( this->getName() + ".bias.grad" );
                zeros( *bias_grad_ );
            }
        }
    };
}