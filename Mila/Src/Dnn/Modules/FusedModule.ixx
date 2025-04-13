module;
#include <memory>

export module Dnn.FusedModule;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.DeviceType;
import Compute.MemoryResource;  
import Compute.CpuMemoryResource;  
import Compute.CudaMemoryResource;  

namespace Mila::Dnn
{
    export
    template<typename TInput, typename TPrecision = TInput, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
        requires ValidTensorTypes<TInput, TPrecision>
    class FusedModule : public Module<TInput, TPrecision, TDeviceType> {
    public:
        explicit FusedModule( std::shared_ptr<FusedOp> op ) : op_( std::move( op ) ) {}

        void forward( const Tensor& input, Tensor& output ) override {
            op_->forward( input, output );
        }

        void build( Device, DType ) override {
            // Fused modules don't require further building
        }

    private:
        std::shared_ptr<FusedOp> op_;
    };
}