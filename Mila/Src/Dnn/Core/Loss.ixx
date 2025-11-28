module;
#include <string>
#include <vector>
#include <type_traits>

export module Dnn.Loss;

import Dnn.Component;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

    /**
     * @brief Abstract base class for neural network loss functions.
     *
     * @tparam TDeviceType Compile-time device identifier for this loss.
     * @tparam TPrecision Data type used for computations.
     *
     * Loss functions compute a scalar loss value given model predictions
     * and target values. They may also provide hooks for optimizing the
     * network graph and configuring reduction modes.
	 */
    export template<Compute::DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Loss : public Component<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;

        virtual ~Loss() = default;

        // Loss-specific compute method (not overriding anything from Module)
        //virtual Tensor compute( const Tensor& prediction, const Tensor& target ) = 0;

        // Network optimization hook (loss-specific)
        //virtual void optimizeNetworkGraph( Network<TDevice>& network )
        //{
        //    // Default: no optimization
        //}

        // Implement Module interface
        //std::vector<Parameter*> parameters() override
        //{
        //    return {};  // Default: no parameters (overridden by losses that have them)
        //}

        //void save( ModelArchive& archive, const std::string& prefix ) const override
        //{
        //    // Save loss state
        //}

        //void load( ModelArchive& archive, const std::string& prefix ) override
        //{
        //    // Load loss state
        //}
    };
}