/**
 * @file ComputePrecision.ixx
 * @brief Defines precision policy for neural network operations.
 */

module;
#include <string_view>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>
#include <string>
#include <sstream>

export module Compute.Precision;

import Dnn.TensorTraits;
import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Controls automatic mixed precision behavior for neural network operations.
     *
     * This lightweight class enables or disables automatic mixed precision (AMP)
     * and provides a basic policy preference, but leaves the actual precision
     * decisions to the underlying libraries and kernels which have sophisticated
     * internal heuristics.
     */
    export class ComputePrecision {
    public:
        /**
         * @brief Basic precision policy preference
         */
        enum class Policy {
            Disabled,    // Do not use mixed precision
            Auto,        // Let kernels and libraries choose the best balance (default)
            Performance, // Prefer performance over accuracy when possible
            Accuracy     // Prioritize accuracy over performance
        };

        /**
         * @brief Construct a ComputePrecision with the specified policy.
         */
        ComputePrecision( Policy policy = Policy::Auto ) : policy_( policy ) {}

        /**
         * @brief Check if automatic mixed precision is enabled.
         */
        bool isEnabled() const { return policy_ != Policy::Disabled; }

        /**
         * @brief Get the current policy.
         */
        Policy getPolicy() const { return policy_; }

        /**
         * @brief Set the policy preference.
         */
        void setPolicy( Policy policy ) { policy_ = policy; }

        /**
         * @brief Convert the precision policy to a string representation.
         *
         * @return std::string String representation of the precision policy.
         */
        std::string toString() const {
            std::ostringstream oss;
            oss << "Precision Policy: ";
            switch ( policy_ ) {
                case Policy::Disabled:
                    oss << "Disabled";
                    break;
                case Policy::Performance:
                    oss << "Performance";
                    break;
                case Policy::Auto:
                    oss << "Auto";
                    break;
                case Policy::Accuracy:
                    oss << "Accuracy";
                    break;
            }
            return oss.str();
        }

    private:
        Policy policy_;
    };
}