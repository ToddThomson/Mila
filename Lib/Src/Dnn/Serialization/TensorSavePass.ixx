/**
 * @file TensorSavePass.ixx
 * @brief The two passes a component's tensors are driven through when a safetensors file is written.
 */

export module Serialization.TensorSavePass;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief Which pass of a two-phase save a component is being driven through.
     *
     * Exists so a component visits its tensors from ONE ordered body rather than two.
     * The writer requires bodies in declaration order, and two separate walks would be
     * free to drift out of agreement with no diagnostic until the file is read back.
     */
    export enum class TensorSavePass
    {
        Declare,  ///< Reserve each tensor's byte range in the header.
        Write     ///< Stream each tensor's bytes, in declaration order.
    };
}
