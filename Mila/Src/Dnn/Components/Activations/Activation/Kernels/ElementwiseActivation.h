/**
 * @file ElementwiseActivation.h
 * @brief Shared elementwise-activation functor library (single source of truth).
 *
 * Each activation is a small POD functor exposing fwd(x) and df(x). The functors
 * are annotated MILA_HD so the identical definitions compile for the MSVC-built
 * CPU operation and the nvcc-built CUDA kernels. This is the one sanctioned
 * preprocessor use, confined to this header's global module fragment, and is the
 * single math source consumed by both CpuElementwiseActivationOp and
 * CudaElementwiseActivationOp (see Specifications/FfnAndMoE.md section 5.1).
 *
 * Design contract:
 *   - POD only: no dependency on Dnn.ActivationType, Tensor, or any module. The
 *     enum->functor bridge lives in the Activation component; function-specific
 *     scalars (LeakyReLU alpha) ride as public members set host-side by the op.
 *   - df(x) recomputes the derivative from the forward input x, giving every
 *     functor a uniform single-argument derivative interface.
 *   - All arithmetic is float; BF16/FP32 callers promote to float before fwd/df.
 */

#pragma once

#include <math.h>

#if defined(__CUDACC__)
    #define MILA_HD __host__ __device__
#else
    #define MILA_HD
#endif

namespace Mila::Dnn::Activations
{
    /// Identity (ActivationType::None): f(x) = x.
    struct Identity
    {
        MILA_HD float fwd( float x ) const { return x; }
        MILA_HD float df( float ) const { return 1.0f; }
    };

    /// GELU, tanh approximation: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).
    struct GeluTanh
    {
        static constexpr float kScale = 0.7978845608f;  // sqrt(2/pi)
        static constexpr float kCoeff = 0.044715f;

        MILA_HD float fwd( float x ) const
        {
            float cube = kCoeff * x * x * x;
            return 0.5f * x * (1.0f + tanhf( kScale * (x + cube) ));
        }

        MILA_HD float df( float x ) const
        {
            float x2 = x * x;
            float arg = kScale * (x + kCoeff * x * x2);
            float t = tanhf( arg );
            float sech2 = 1.0f - t * t;
            float darg = kScale * (1.0f + 3.0f * kCoeff * x2);
            return 0.5f * (1.0f + t) + 0.5f * x * sech2 * darg;
        }
    };

    /// SiLU / Swish: x * sigmoid(x).
    struct Silu
    {
        MILA_HD float fwd( float x ) const
        {
            float s = 1.0f / (1.0f + expf( -x ));
            return x * s;
        }

        MILA_HD float df( float x ) const
        {
            float s = 1.0f / (1.0f + expf( -x ));
            return s * (1.0f + x * (1.0f - s));
        }
    };

    /// ReLU: max(0, x).
    struct Relu
    {
        MILA_HD float fwd( float x ) const { return x > 0.0f ? x : 0.0f; }
        MILA_HD float df( float x ) const { return x > 0.0f ? 1.0f : 0.0f; }
    };

    /// Hyperbolic tangent.
    struct Tanh
    {
        MILA_HD float fwd( float x ) const { return tanhf( x ); }
        MILA_HD float df( float x ) const
        {
            float t = tanhf( x );
            return 1.0f - t * t;
        }
    };

    /// Logistic sigmoid: 1 / (1 + exp(-x)).
    struct Sigmoid
    {
        MILA_HD float fwd( float x ) const { return 1.0f / (1.0f + expf( -x )); }
        MILA_HD float df( float x ) const
        {
            float s = 1.0f / (1.0f + expf( -x ));
            return s * (1.0f - s);
        }
    };

    /// Leaky ReLU: x for x > 0, alpha*x otherwise. alpha carried by value.
    struct LeakyRelu
    {
        float alpha = 0.01f;

        MILA_HD float fwd( float x ) const { return x > 0.0f ? x : alpha * x; }
        MILA_HD float df( float x ) const { return x > 0.0f ? 1.0f : alpha; }
    };

    /// Mish: x * tanh(softplus(x)).
    struct Mish
    {
        // Numerically stable softplus: for large x, log(1 + e^x) -> x.
        MILA_HD static float softplus( float x )
        {
            return x > 20.0f ? x : log1pf( expf( x ) );
        }

        MILA_HD float fwd( float x ) const
        {
            return x * tanhf( softplus( x ) );
        }

        MILA_HD float df( float x ) const
        {
            float sp = softplus( x );
            float t = tanhf( sp );
            float s = 1.0f / (1.0f + expf( -x ));
            return t + x * (1.0f - t * t) * s;
        }
    };
}
