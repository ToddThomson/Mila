/**
 * @file Module.ixx
 * @brief Defines the base Module class for the Mila deep neural network framework.
 *
 * The Module class provides a unified interface for neural network layers with proper
 * support for abstract tensor data types, scalar tensors, and device-agnostic operations.
 */

module;
#include <string>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <type_traits>
#include <sstream>
#include <format>

export module Dnn.Module;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ConfigurationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.CpuExecutionContext;
import Compute.CudaExecutionContext;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.Precision;
import Serialization.ModelArchive;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    export template<DeviceType TDeviceType>
    class Module
    {
    public:
        virtual ~Module() = default;

        // ====================================================================
        // Core Interface (Pure Virtual)
        // ====================================================================

        virtual void forward( const ITensor& input, ITensor& output ) = 0;
        
        virtual void backward( 
            const ITensor& input,
            const ITensor& output_grad,
            ITensor& input_grad ) = 0;

        virtual size_t parameterCount() const = 0;

        virtual void save( ModelArchive& archive ) const = 0;
        virtual void load( ModelArchive& archive ) = 0;

        virtual std::string toString() const = 0;

        // ====================================================================
        // State and Configuration (Pure Virtual)
        // ====================================================================

        virtual void setTraining( bool is_training ) = 0;
        virtual bool isTraining() const = 0;

        virtual std::string getName() const = 0;
        
        // ====================================================================
        // Device Information (Static - constexpr)
        // ====================================================================

        static constexpr DeviceType getDeviceType() {
            return TDeviceType;
        }

        // ====================================================================
        // Operators
        // ====================================================================

        friend std::ostream& operator<<( std::ostream& os, const Module& module ) {
            os << module.toString();
            return os;
        }
    };
}