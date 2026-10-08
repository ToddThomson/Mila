/**
 * @file GatedResidual.Config.ixx
 * @brief Configuration for the gated residual (hyper-connections): stream count, width, mixer rank.
 *
 * Qwen 4 carries its residual as n streams of width H and reads each sublayer's input through one of these.
 * See Specifications/Qwen4.md section 3.1.
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>
#include <format>
#include <utility>
#include <cstdint>

export module Dnn.Components.GatedResidualConfig;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    export class GatedResidualConfig : public ComponentConfig
    {
    public:
        /**
         * @param model_dim H, the width of one stream and of the sublayer's input.
         * @param streams   n, the stream count (Qwen 4: hc_count). The residual state is n * H wide.
         * @param rank      The low-rank mixer's inner width (Qwen 4: hc_lowrank).
         */
        GatedResidualConfig( dim_t model_dim, dim_t streams, dim_t rank )
            : model_dim_( model_dim ), streams_( streams ), rank_( rank )
        {
            validate();
        }

        /**
         * @brief Whether the residual also injects a sublayer's output back into the streams. Default: true.
         *
         * False is the read-only form that ends the network, reducing the streams to the H-wide input of
         * the language-model head; it has no inject projection.
         */
        template<typename Self>
        decltype(auto) withInjection( this Self&& self, bool has_injection )
        {
            self.has_injection_ = has_injection;

            return std::forward<Self>( self );
        }

        /// Epsilon of the grouped stream norm (Qwen 4: rms_norm_eps).
        template<typename Self>
        decltype(auto) withEpsilon( this Self&& self, float epsilon )
        {
            self.epsilon_ = epsilon;
            self.validate();

            return std::forward<Self>( self );
        }

        dim_t getModelDim() const noexcept { return model_dim_; }
        dim_t getStreams() const noexcept { return streams_; }
        dim_t getRank() const noexcept { return rank_; }
        bool hasInjection() const noexcept { return has_injection_; }
        float getEpsilon() const noexcept { return epsilon_; }

        /// n * H, the width of the residual state.
        dim_t getStreamWidth() const noexcept { return streams_ * model_dim_; }

        void validate() const override
        {
            if ( model_dim_ <= 0 )
            {
                throw std::invalid_argument( "GatedResidualConfig: model_dim must be > 0" );
            }

            // One stream is a plain residual; the reference refuses it too.
            if ( streams_ <= 1 )
            {
                throw std::invalid_argument( std::format( "GatedResidualConfig: streams must be > 1, got {}", streams_ ) );
            }

            if ( rank_ <= 0 )
            {
                throw std::invalid_argument( "GatedResidualConfig: rank must be > 0" );
            }

            if ( !( epsilon_ > 0.0f ) )
            {
                throw std::invalid_argument( "GatedResidualConfig: epsilon must be > 0" );
            }
        }

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "model_dim", static_cast<int64_t>( model_dim_ ) )
                .set( "streams", static_cast<int64_t>( streams_ ) )
                .set( "rank", static_cast<int64_t>( rank_ ) )
                .set( "has_injection", has_injection_ )
                .set( "epsilon", epsilon_ );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "model_dim" ) )
            {
                model_dim_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "streams" ) )
            {
                streams_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "rank" ) )
            {
                rank_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetBool( "has_injection" ) )
            {
                has_injection_ = *v;
            }

            if ( auto v = meta.tryGetFloat( "epsilon" ) )
            {
                epsilon_ = *v;
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "GatedResidualConfig( model_dim=" << model_dim_ << ", streams=" << streams_ << ", rank=" << rank_
                << ", has_injection=" << ( has_injection_ ? "true" : "false" ) << ", epsilon=" << epsilon_ << " )";

            return oss.str();
        }

    private:
        dim_t model_dim_{ 0 };
        dim_t streams_{ 0 };
        dim_t rank_{ 0 };
        bool has_injection_{ true };
        float epsilon_{ 1e-6f };
    };
}
