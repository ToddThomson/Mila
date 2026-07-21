/**
 * @file Decoder.ixx
 * @brief Base interface for Mila decoders.
 */

module;
#include <memory>

export module Dnn.DecoderBase;

import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Decoder
    {
    public:
        virtual ~Decoder() = default;

        virtual void decode( const ITensor logits ) = 0;
    };
}
