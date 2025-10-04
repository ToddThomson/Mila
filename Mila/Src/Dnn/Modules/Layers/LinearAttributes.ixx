/**
 * @file LinearAttributes.ixx
 * @brief Defines tensor attributes for linear (fully-connected) operations
 */

module;
#include <unordered_map>
#include <memory>

export module Dnn.Modules.Linear:Attributes;

import Dnn.Attributes;
import Dnn.ITensor;

namespace Mila::Dnn::Modules
{
    /**
     * @brief Attributes for linear/fully-connected operations
     *
     * Provides typed access to tensors used in linear operations including
     * weights, bias, and their gradients. Also tracks intermediate values
     * needed during the backward pass.
     */
    export class LinearAttributes : public Attributes<LinearAttributes> {
    public:
        /**
         * @brief Tensor roles in linear operations
         *
         * Identifies the purpose of each tensor in the linear operation
         */
        enum class InputNames {
			X,              // Input tensor (batch_size × input_dim)
            Weight,         // Weight matrix (output_dim × input_dim)
            Bias            // Bias vector (output_dim)
        };

        std::unordered_map<InputNames, std::shared_ptr<ITensor>> inputs_;

        enum class OutputNames {
			Y,              // Output tensor (batch_size × output_dim)
        };

        std::unordered_map<OutputNames, std::shared_ptr<ITensor>> outputs_;
    };
}