/**
 * @file CpuRmsNormOp.ixx
 * @brief CPU implementation of RMS normalization (FP32).
 *
 * The reference loop behind RmsNorm<Cpu>, mirroring CudaRmsNormOp's partitioning and its unit offset.
 */

module;
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

export module Compute.CpuRmsNormOp;

import Dnn.Components.RmsNormConfig;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.Component;
import Compute.DeviceType;
import Compute.IExecutionContext;
import Compute.OperationType;
import Compute.OperationBase;

namespace Mila::Dnn::Compute
{
    using namespace Mila::Dnn;

    /**
     * @brief RMS normalization over one axis, with an optional affine transform.
     *
     *   rstd = 1 / sqrt( mean(x^2) + epsilon )
     *   y    = x * rstd * ( weight + unit_offset ) + bias
     *
     * The normalization axis is fixed at build; the slice count follows each call's input, bounded
     * by the build, because a component built at the prefill width is called with narrower inputs.
     */
    export class CpuRmsNormOp : public Operation<DeviceType::Cpu, TensorDataType::FP32>
    {
    public:
        using OperationBaseType = Operation<DeviceType::Cpu, TensorDataType::FP32>;

        CpuRmsNormOp( IExecutionContext* context, const RmsNormConfig& config )
            : context_( context ), config_( config )
        {
            if ( !context_ )
            {
                throw std::runtime_error( "CpuRmsNormOp requires a CPU execution context" );
            }

            config_.validate();
        }

        void setParameters( ITensor* weight, ITensor* bias ) override
        {
            if ( !weight )
            {
                throw std::invalid_argument( "CpuRmsNormOp::setParameters - weight parameter is required" );
            }

            weight_ = static_cast<const float*>( weight->rawData() );

            if ( config_.hasBias() )
            {
                if ( !bias )
                {
                    throw std::invalid_argument( "CpuRmsNormOp::setParameters - bias parameter expected but null was provided" );
                }

                bias_ = static_cast<const float*>( bias->rawData() );
            }
            else
            {
                bias_ = nullptr;
            }
        }

        void setGradients( ITensor* weight_grad, ITensor* bias_grad ) override
        {
            if ( !weight_grad )
            {
                throw std::invalid_argument( "CpuRmsNormOp::setGradients - weight gradient is required" );
            }

            weight_grad_ = static_cast<float*>( weight_grad->rawData() );

            if ( config_.hasBias() )
            {
                if ( !bias_grad )
                {
                    throw std::invalid_argument( "CpuRmsNormOp::setGradients - bias gradient expected but null was provided" );
                }

                bias_grad_ = static_cast<float*>( bias_grad->rawData() );
            }
            else
            {
                bias_grad_ = nullptr;
            }
        }

        void clearGradients() noexcept override
        {
            weight_grad_ = nullptr;
            bias_grad_ = nullptr;
        }

        void build( const BuildContext& build_context ) override
        {
            if ( weight_ == nullptr )
            {
                throw std::runtime_error( "CpuRmsNormOp::build requires parameters bound via setParameters() before build()" );
            }

            if ( config_.hasBias() && bias_ == nullptr )
            {
                throw std::runtime_error( "CpuRmsNormOp::build - bias expected by config but not bound via setParameters()" );
            }

            const auto& shape = build_context.inputShape();
            const int64_t ndim = static_cast<int64_t>( shape.size() );

            int64_t axis = -1;

            if ( !config_.getNormalizedShape().empty() )
            {
                const auto& normalized_shape = config_.getNormalizedShape();

                if ( shape.size() < normalized_shape.size() )
                {
                    throw std::invalid_argument( "CpuRmsNormOp::build - input rank is less than normalized_shape rank" );
                }

                const size_t offset = shape.size() - normalized_shape.size();

                for ( size_t i = 0; i < normalized_shape.size(); ++i )
                {
                    if ( shape[ offset + i ] != normalized_shape[ i ] )
                    {
                        throw std::invalid_argument( "CpuRmsNormOp::build - input trailing dimensions don't match normalized_shape" );
                    }
                }

                axis = static_cast<int64_t>( offset );
            }
            else if ( config_.getAxis().has_value() )
            {
                axis = config_.getAxis().value();
            }
            else
            {
                throw std::invalid_argument( "CpuRmsNormOp::build - configuration must specify normalized_shape or axis before build()" );
            }

            if ( axis < 0 )
            {
                axis += ndim;
            }

            if ( axis < 0 || axis >= ndim )
            {
                throw std::invalid_argument( "CpuRmsNormOp::build - computed axis out of range" );
            }

            normalized_axis_ = axis;
            normalized_size_ = static_cast<int64_t>( shape[ axis ] );

            const Geometry geometry = geometryFor( shape );
            rstd_.assign( static_cast<size_t>( geometry.outer * geometry.inner ), 0.0f );

            OperationBaseType::build( build_context );
        }

        void forward( const ITensor& input, ITensor& output ) const
        {
            const Geometry geometry = runtimeGeometry( input.shape(), "forward" );
            const float offset = config_.getUnitOffset();
            const double epsilon = static_cast<double>( config_.getEpsilon() );

            const auto* x = static_cast<const float*>( input.rawData() );
            auto* y = static_cast<float*>( output.rawData() );

            for ( int64_t outer = 0; outer < geometry.outer; ++outer )
            {
                for ( int64_t inner = 0; inner < geometry.inner; ++inner )
                {
                    const int64_t base = outer * normalized_size_ * geometry.inner + inner;

                    double sum_of_squares = 0.0;

                    for ( int64_t i = 0; i < normalized_size_; ++i )
                    {
                        const double value = x[ base + i * geometry.inner ];
                        sum_of_squares += value * value;
                    }

                    const double rstd = 1.0 / std::sqrt( sum_of_squares / static_cast<double>( normalized_size_ ) + epsilon );
                    rstd_[ static_cast<size_t>( outer * geometry.inner + inner ) ] = static_cast<float>( rstd );

                    for ( int64_t i = 0; i < normalized_size_; ++i )
                    {
                        double normalized = x[ base + i * geometry.inner ] * rstd * ( static_cast<double>( weight_[ i ] ) + offset );

                        if ( bias_ )
                        {
                            normalized += bias_[ i ];
                        }

                        y[ base + i * geometry.inner ] = static_cast<float>( normalized );
                    }
                }
            }
        }

        /**
         * @brief Input gradient, and parameter gradients when bound. Accumulates into all three.
         *
         * With g_i = (weight_i + unit_offset) * dy_i and n the normalized size:
         *   dx_i = rstd * g_i - x_i * rstd^3 * sum_j( g_j * x_j ) / n
         * Reads the rstd forward() stored, so it must follow forward() on the same input.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad ) const
        {
            const Geometry geometry = runtimeGeometry( input.shape(), "backward" );
            const float offset = config_.getUnitOffset();

            const auto* x = static_cast<const float*>( input.rawData() );
            const auto* dy = static_cast<const float*>( output_grad.rawData() );
            auto* dx = static_cast<float*>( input_grad.rawData() );

            for ( int64_t outer = 0; outer < geometry.outer; ++outer )
            {
                for ( int64_t inner = 0; inner < geometry.inner; ++inner )
                {
                    const int64_t base = outer * normalized_size_ * geometry.inner + inner;
                    const double rstd = rstd_[ static_cast<size_t>( outer * geometry.inner + inner ) ];

                    double gradient_dot_input = 0.0;

                    for ( int64_t i = 0; i < normalized_size_; ++i )
                    {
                        const int64_t at = base + i * geometry.inner;
                        gradient_dot_input += ( static_cast<double>( weight_[ i ] ) + offset ) * dy[ at ] * x[ at ];
                    }

                    const double correction = rstd * rstd * rstd * gradient_dot_input / static_cast<double>( normalized_size_ );

                    for ( int64_t i = 0; i < normalized_size_; ++i )
                    {
                        const int64_t at = base + i * geometry.inner;
                        const double gradient = ( static_cast<double>( weight_[ i ] ) + offset ) * dy[ at ];

                        dx[ at ] += static_cast<float>( rstd * gradient - x[ at ] * correction );

                        if ( weight_grad_ )
                        {
                            weight_grad_[ i ] += static_cast<float>( x[ at ] * rstd * dy[ at ] );
                        }

                        if ( bias_grad_ )
                        {
                            bias_grad_[ i ] += dy[ at ];
                        }
                    }
                }
            }
        }

        OperationType getOperationType() const override
        {
            return OperationType::RmsNormOp;
        }

        std::string getName() const override
        {
            return "Cpu::RmsNormOp";
        }

    private:
        struct Geometry
        {
            int64_t outer{ 1 };
            int64_t inner{ 1 };
        };

        Geometry geometryFor( const shape_t& shape ) const
        {
            Geometry geometry;

            for ( int64_t i = 0; i < normalized_axis_; ++i )
            {
                geometry.outer *= static_cast<int64_t>( shape[ i ] );
            }

            for ( int64_t i = normalized_axis_ + 1; i < static_cast<int64_t>( shape.size() ); ++i )
            {
                geometry.inner *= static_cast<int64_t>( shape[ i ] );
            }

            return geometry;
        }

        Geometry runtimeGeometry( const shape_t& shape, const char* caller ) const
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( std::string( "CpuRmsNormOp::" ) + caller + " - operation must be built first" );
            }

            if ( normalized_axis_ >= static_cast<int64_t>( shape.size() )
                 || static_cast<int64_t>( shape[ normalized_axis_ ] ) != normalized_size_ )
            {
                throw std::runtime_error( std::string( "CpuRmsNormOp::" ) + caller
                    + " - input shape is incompatible with the built normalization axis" );
            }

            const Geometry geometry = geometryFor( shape );

            if ( static_cast<size_t>( geometry.outer * geometry.inner ) > rstd_.size() )
            {
                throw std::runtime_error( std::string( "CpuRmsNormOp::" ) + caller
                    + " - runtime slice count exceeds the built maximum" );
            }

            return geometry;
        }

        IExecutionContext* context_;
        RmsNormConfig config_;

        const float* weight_{ nullptr };
        const float* bias_{ nullptr };
        float* weight_grad_{ nullptr };
        float* bias_grad_{ nullptr };

        int64_t normalized_axis_{ -1 };
        int64_t normalized_size_{ 0 };

        // Written by the const forward() and read by backward(); the op holds no other state.
        mutable std::vector<float> rstd_;
    };
}
