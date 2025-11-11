module;
#include <memory>

export module Dnn.FusedModule;

import Dnn.Module;
import Dnn.Tensor;
import Dnn.TensorTraits;
import Compute.DeviceType;
import Compute.MemoryResource;  
import Compute.CpuMemoryResource;  
import Compute.CudaDeviceMemoryResource;  

namespace Mila::Dnn
{
	// REVIEW: Not required? To be removed. Commented out for now.
    // 
    //export template<typename TPrecision, typename TInput = TPrecision, Compute::DeviceType TDeviceType = Compute::DeviceType::Cuda>
    //    requires ValidTensorTypes<TPrecision, TInput>
    //class FusedModule : public Module<TPrecision, TInput, TDeviceType> {
    //public:
    //    explicit FusedModule( std::shared_ptr<FusedOp> op ) : op_( std::move( op ) ) {}

    //    void forward( const Tensor& input, Tensor& output ) override {
    //        op_->forward( input, output );
    //    }

    //    void build( Device, DType ) override {
    //        // Fused modules don't require further building
    //    }

    //private:
    //    std::shared_ptr<FusedOp> op_;
    //};
}