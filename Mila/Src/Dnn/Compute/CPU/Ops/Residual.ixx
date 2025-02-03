module;

#ifdef USE_OMP
#include <omp.h>
#endif

export module Compute.CpuResidual;

import Compute.OperationBase;
import Compute.DeviceType;
import Compute.OperationType;

namespace Mila::Dnn::Compute
{
    export
    template<typename T>
    class CpuResidual :public OperationBase<T> {
    public:
        CpuResidual() : OperationBase<T>( DeviceType::kCpu, OperationType::kResidual ) {}

        void forward( float* out, float* inp1, float* inp2, int N ) {
            for ( int i = 0; i < N; i++ ) {
                out[ i ] = inp1[ i ] + inp2[ i ];
            }
        }

        void backward( float* dinp1, float* dinp2, float* dout, int N ) {
            for ( int i = 0; i < N; i++ ) {
                dinp1[ i ] += dout[ i ];
                dinp2[ i ] += dout[ i ];
            }
        }
    };
}