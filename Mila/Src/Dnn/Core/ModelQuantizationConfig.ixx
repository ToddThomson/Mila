/**
 * @file QuantizationConfig.ixx
 * @brief Model-wide weight quantization policy for Mila DNN models.
 *
 * QuantizationConfig is a lightweight value type that carries the weight
 * storage precision policy from LlamaModelConfig through BuildContext
 * to the components that consume it (currently Linear only).
 *
 * It is intentionally decoupled from ComponentConfig and ComputePrecision::Policy:
 *  - ComputePrecision::Policy governs accumulator/algorithm selection (cuBLASLt heuristics)
 *  - QuantizationConfig governs weight storage dtype and scale allocation
 *
 * Construction is via named factory methods only. The default is none().
 *
 * FP4 is stubbed for forward compatibility but not yet implemented.
 * Attempting to use fp4() at build time will throw std::runtime_error.
 */

module;
#include <string>
#include <stdexcept>

// DEPRECATED: No longer used

export module Dnn.QuantizationConfig_OLD;
//
//namespace Mila::Dnn
//{
//    /**
//     * @brief Weight quantization policy for a deployable model.
//     *
//     * Specifies how weight tensors are stored and scaled at load time.
//     * Applied uniformly across all quantizable components (Linear).
//     *
//     * Policies:
//     *   None      - No quantization. Weights stored in TComputePrecision (e.g. BF16).
//     *   FP8_E4M3  - Weights quantized to FP8_E4M3 at load time with FP32
//     *               per-channel scales. Requires SM >= 8.9 (RTX 40xx).
//     *   FP4       - Reserved for future RTX 50xx tensor core support.
//     *               Not yet implemented; throws at build time if used.
//     */
//    export class QuantizationConfig
//    {
//    public:
//
//        enum class Policy
//        {
//            None,
//            FP8_E4M3,
//            FP4       // stub — not yet implemented
//        };
//
//        // ====================================================================
//        // Factory methods
//        // ====================================================================
//
//        /**
//         * @brief No quantization — weights stored in model compute precision.
//         *
//         * Default policy. Linear uses TComputePrecision for both activations
//         * and weights. No scale tensors are allocated.
//         */
//        static QuantizationConfig none()
//        {
//            return QuantizationConfig( Policy::None );
//        }
//
//        /**
//         * @brief FP8_E4M3 weight quantization with FP32 per-channel scales.
//         *
//         * Weights are quantized from BF16 to FP8_E4M3 at model load time.
//         * A FP32 per-channel scale vector is computed and stored alongside
//         * the quantized weight tensor. Scale computation runs once on the
//         * forward pass hot path.
//         *
//         * Requires CUDA SM >= 8.9 (RTX 40xx / Ada Lovelace).
//         * CudaLinearOp::supportsCuBLASLt() enforces the SM requirement at
//         * build time.
//         */
//        static QuantizationConfig fp8()
//        {
//            return QuantizationConfig( Policy::FP8_E4M3 );
//        }
//
//        /**
//         * @brief FP4 weight quantization — reserved for future RTX 50xx support.
//         *
//         * Not yet implemented. Returns a QuantizationConfig that will throw
//         * std::runtime_error if passed to any component's build().
//         */
//        static QuantizationConfig fp4()
//        {
//            return QuantizationConfig( Policy::FP4 );
//        }
//
//        // ====================================================================
//        // Queries
//        // ====================================================================
//
//        Policy getPolicy() const noexcept
//        {
//            return policy_;
//        }
//
//        bool isNone() const noexcept
//        {
//            return policy_ == Policy::None;
//        }
//
//        bool isFp8() const noexcept
//        {
//            return policy_ == Policy::FP8_E4M3;
//        }
//
//        bool isFp4() const noexcept
//        {
//            return policy_ == Policy::FP4;
//        }
//
//        /**
//         * @brief True when any quantization is active.
//         *
//         * Convenience for components that branch on quantized vs non-quantized
//         * without caring about the specific policy.
//         */
//        bool isQuantized() const noexcept
//        {
//            return policy_ != Policy::None;
//        }
//
//        /**
//         * @brief Assert that this policy is safe to use at build time.
//         *
//         * Called by Linear::build() before acting on the policy.
//         * Throws for any policy that is stubbed but not yet implemented.
//         *
//         * @throws std::runtime_error if the policy is FP4.
//         */
//        void assertSupported() const
//        {
//            if ( policy_ == Policy::FP4 )
//            {
//                throw std::runtime_error(
//                    "QuantizationConfig: FP4 quantization is not yet implemented. "
//                    "Requires RTX 50xx (Blackwell) tensor core support." );
//            }
//        }
//
//        // ====================================================================
//        // Diagnostics
//        // ====================================================================
//
//        std::string toString() const
//        {
//            switch ( policy_ )
//            {
//                case Policy::None:     return "Quantization: None";
//                case Policy::FP8_E4M3: return "Quantization: FP8_E4M3 (per-channel FP32 scales)";
//                case Policy::FP4:      return "Quantization: FP4 (not yet implemented)";
//                default:               return "Quantization: Unknown";
//            }
//        }
//
//    private:
//
//        explicit QuantizationConfig( Policy policy )
//            : policy_( policy )
//        {
//        }
//
//        Policy policy_{ Policy::None };
//    };
//}