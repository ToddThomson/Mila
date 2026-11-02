/**
 * @file TensorOps.Traits.ixx
 * @brief Maps a device type to its concrete tensor operations class.
 */

export module Dnn.TensorOpsTraits;

import Compute.DeviceType;

namespace Mila::Dnn
{
    /**
     * @brief Device-to-backend map behind the TensorOps alias template.
     *
     * Public because TensorOps is: a consumer instantiating a device-neutral entry point
     * resolves the alias through this map. Adding a backend means specializing it.
     *
     * Each backend specializes this in its own module with a single `type` alias, for the
     * same reason as Compute::ExecutionContextTraits: MSVC 14.51 will not complete a
     * merely reachable explicit specialization when the dereference is dependent, and the
     * device-neutral entry points in Dnn.TensorOps all dispatch through a dependent
     * `TensorOps<TMemoryResource::device_type>`. Holding the specializations here lets
     * Dnn.TensorOps publish the map without publishing the backends.
     *
     * @tparam TDevice Compute device type to map.
     */
    export template<Compute::DeviceType TDevice>
    struct TensorOpsTraits;
}
