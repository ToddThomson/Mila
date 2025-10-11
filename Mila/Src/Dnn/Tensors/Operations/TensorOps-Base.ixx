export module Dnn.TensorOps.Base;

import Compute.DeviceType;

namespace Mila::Dnn
{
    export template<Compute::DeviceType TDevice> struct TensorOps;
}