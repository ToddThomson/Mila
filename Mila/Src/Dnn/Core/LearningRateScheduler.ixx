/**
 * @file LearningRateScheduler.ixx
 * @brief Learning-rate scheduler base and common concrete schedules.
 *
 * Provides a minimal abstract scheduler API and three implementations:
 * Constant, Linear decay, and Cosine annealing.
 */

module;
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <stdexcept>
#include <format>

export module Dnn.LearningRateScheduler;

namespace Mila::Dnn
{
    /**
     * @brief Abstract base for learning-rate schedulers.
     *
     * Implementations compute a scalar learning rate for a given step index.
     * The step argument is a zero-based iteration/step count. Implementations
     * must be thread-safe for concurrent const calls.
     */
    export class LearningRateScheduler
    {
    public:

        virtual ~LearningRateScheduler() = default;

        /**
         * @brief Get the learning rate for the provided (zero-based) step.
         *
         * @param step Zero-based step index.
         * @return Scalar learning rate to use at this step.
         */
        virtual double getLearningRate( std::size_t step ) const = 0;

        /**
         * @brief Human-readable description of the scheduler.
         */
        virtual std::string toString() const = 0;
    };

    /**
     * @brief Constant learning-rate scheduler.
     *
     * Always returns the same learning rate.
     */
    export class ConstantLRScheduler final : public LearningRateScheduler
    {
    public:

        explicit ConstantLRScheduler( double lr )
            : lr_( lr )
        {
            if ( lr_ <= 0.0 )
            {
                throw std::invalid_argument( "ConstantLRScheduler: lr must be > 0" );
            }
        }

        double getLearningRate( std::size_t /*step*/ ) const override
        {
            return lr_;
        }

        std::string toString() const override
        {
            return std::format( "ConstantLRScheduler(lr={:.6g})", lr_ );
        }

    private:

        double lr_;
    };

    /**
     * @brief Linear decay scheduler.
     *
     * Linearly interpolates from `initial_lr` at step 0 to `final_lr` at
     * `total_steps`. For steps >= total_steps the scheduler returns final_lr.
     *
     * Preconditions:
     * - total_steps > 0
     * - initial_lr >= 0, final_lr >= 0
     */
    export class LinearLRScheduler final : public LearningRateScheduler
    {
    public:

        LinearLRScheduler( double initial_lr, double final_lr, std::size_t total_steps )
            : initial_lr_( initial_lr ), final_lr_( final_lr ), total_steps_( total_steps )
        {
            if ( total_steps_ == 0 )
            {
                throw std::invalid_argument( "LinearLRScheduler: total_steps must be > 0" );
            }

            if ( initial_lr_ < 0.0 || final_lr_ < 0.0 )
            {
                throw std::invalid_argument( "LinearLRScheduler: learning rates must be >= 0" );
            }
        }

        double getLearningRate( std::size_t step ) const override
        {
            if ( step >= total_steps_ )
            {
                return final_lr_;
            }

            double t = static_cast<double>( step ) / static_cast<double>( total_steps_ );
            double lr = initial_lr_ + ( final_lr_ - initial_lr_ ) * t;

            return lr;
        }

        std::string toString() const override
        {
            return std::format( "LinearLRScheduler(initial={:.6g}, final={:.6g}, steps={})",
                initial_lr_, final_lr_, total_steps_ );
        }

    private:

        double initial_lr_;
        double final_lr_;
        std::size_t total_steps_;
    };

    /**
     * @brief Cosine annealing scheduler.
     *
     * Uses cosine schedule from `initial_lr` to `final_lr` over `total_steps`.
     * For step >= total_steps the scheduler returns final_lr.
     *
     * Formula (t in [0,1]): lr = final + 0.5*(initial - final)*(1 + cos(pi * t))
     *
     * Preconditions:
     * - total_steps > 0
     * - initial_lr >= 0, final_lr >= 0
     */
    export class CosineLRScheduler final : public LearningRateScheduler
    {
    public:

        CosineLRScheduler( double initial_lr, double final_lr, std::size_t total_steps )
            : initial_lr_( initial_lr ), final_lr_( final_lr ), total_steps_( total_steps )
        {
            if ( total_steps_ == 0 )
            {
                throw std::invalid_argument( "CosineLRScheduler: total_steps must be > 0" );
            }

            if ( initial_lr_ < 0.0 || final_lr_ < 0.0 )
            {
                throw std::invalid_argument( "CosineLRScheduler: learning rates must be >= 0" );
            }
        }

        double getLearningRate( std::size_t step ) const override
        {
            if ( step >= total_steps_ )
            {
                return final_lr_;
            }

            double t = static_cast<double>( step ) / static_cast<double>( total_steps_ );
            double cos_arg = M_PI * t;
            double factor = 0.5 * ( 1.0 + std::cos( cos_arg ) );
            double lr = final_lr_ + ( initial_lr_ - final_lr_ ) * factor;

            return lr;
        }

        std::string toString() const override
        {
            return std::format( "CosineLRScheduler(initial={:.6g}, final={:.6g}, steps={})",
                initial_lr_, final_lr_, total_steps_ );
        }

    private:

        double initial_lr_;
        double final_lr_;
        std::size_t total_steps_;
    };
}