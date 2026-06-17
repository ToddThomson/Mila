/**
 * @file GradientCheck.h
 * @brief Reusable finite-difference gradient verifier (see Specifications/Testing.md).
 *
 * The authored Mila test suite was forward-only: inference validated forward
 * passes against HuggingFace, so every component backward() the training samples
 * drive had zero coverage. This header is the shared archetype that closes that
 * gap without re-deriving an analytic reference per component.
 *
 * The check is black-box. For a fixed upstream gradient g it forms the scalar
 * loss L = sum_j output[j] * g[j] and verifies, by central differences, that the
 * analytic gradient backward() produces for a buffer (the input, or a parameter
 * tensor) matches dL/dbuffer. This works for any component because
 *   dL/dx_i = sum_j g[j] * d output[j] / d x_i = (J^T g)_i,
 * which is exactly what backward() computes from the same g. The component's own
 * forward() is the only source of truth, so the verifier carries no per-component
 * math and is agnostic to the forward/backward signature (the caller adapts it in
 * a lambda).
 */

#pragma once

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace Mila::Tests::Common
{
    /**
     * @brief Central-difference numeric gradient of the scalar loss L = sum(output * upstream_gradient).
     *
     * @param perturbable       The buffer to differentiate with respect to (the input tensor's
     *                          data, or a parameter tensor's rawData). Mutated in place during the
     *                          probe and restored to its original values before returning.
     * @param perturbable_size  Element count of @p perturbable.
     * @param upstream_gradient The fixed output gradient g defining the scalar loss; length @p output_size.
     * @param output_size       Element count of the forward output.
     * @param evaluateOutput    Callable () -> const float* that runs the component's forward pass
     *                          against the CURRENT contents of @p perturbable and returns a pointer
     *                          to the freshly computed output (length @p output_size).
     * @param epsilon           Half-step for the central difference. 1e-2f suits FP32: large enough
     *                          to dominate cancellation roundoff, small enough that the O(eps^2)
     *                          truncation error stays under a 1e-2 tolerance.
     * @return dL/d perturbable[i] for each element i -- compare against the analytic gradient.
     */
    template <typename EvaluateOutput>
    std::vector<float> centralDifferenceGradient(
        float* perturbable,
        std::size_t perturbable_size,
        const float* upstream_gradient,
        std::size_t output_size,
        EvaluateOutput&& evaluateOutput,
        float epsilon = 1e-2f )
    {
        std::vector<float> numeric( perturbable_size, 0.0f );

        auto scalarLoss = [&]() -> double
        {
            const float* output = evaluateOutput();
            double loss = 0.0;

            for ( std::size_t j = 0; j < output_size; ++j )
            {
                loss += static_cast<double>( output[ j ] ) * static_cast<double>( upstream_gradient[ j ] );
            }

            return loss;
        };

        for ( std::size_t i = 0; i < perturbable_size; ++i )
        {
            const float original = perturbable[ i ];

            perturbable[ i ] = original + epsilon;
            const double loss_plus = scalarLoss();

            perturbable[ i ] = original - epsilon;
            const double loss_minus = scalarLoss();

            perturbable[ i ] = original;
            numeric[ i ] = static_cast<float>( ( loss_plus - loss_minus ) / ( 2.0 * static_cast<double>( epsilon ) ) );
        }

        return numeric;
    }

    /**
     * @brief Assert an analytic gradient matches its numeric counterpart elementwise.
     *
     * Uses a combined absolute/relative tolerance: the relative term is essential because
     * finite-difference magnitudes vary by orders of magnitude across a single layer, so a
     * flat absolute tolerance is either too loose for small entries or too tight for large ones.
     *
     * @param analytic            Analytic gradient (component backward / getGradients); length @p numeric.size().
     * @param numeric             Numeric gradient from centralDifferenceGradient.
     * @param absolute_tolerance  Floor tolerance for near-zero entries (FP32 ~1e-2).
     * @param relative_tolerance  Fraction-of-magnitude tolerance (FP32 ~1e-2).
     * @param label               Identifier for the failure message (e.g. "LayerNorm dW").
     */
    inline void expectGradientsClose(
        const float* analytic,
        const std::vector<float>& numeric,
        float absolute_tolerance,
        float relative_tolerance,
        const std::string& label )
    {
        for ( std::size_t i = 0; i < numeric.size(); ++i )
        {
            const float tolerance = absolute_tolerance + relative_tolerance * std::fabs( numeric[ i ] );

            EXPECT_NEAR( analytic[ i ], numeric[ i ], tolerance )
                << label << ": gradient mismatch at index " << i;
        }
    }
}
