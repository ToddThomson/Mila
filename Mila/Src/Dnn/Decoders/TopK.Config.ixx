/**
 * @file TopK.Config.ixx
 * @brief TopK decoderconfiguration.
 *
 * Fluent setters, validation and metadata serialization for TopK hyperparameters.
 */

module;
#include <string>
#include <stdexcept>
#include <sstream>
#include <utility>

export module Dnn.Decoders.TopKDecoder:Config;

namespace Mila::Dnn
{
    export class TopKConfig
    {
    public:
        template<typename Self>
        decltype( auto ) withK( this Self&& self, size_t k ) noexcept
        {
            self.k_ = k;
            return std::forward<Self>( self );
        }
        /**
         * @brief Validate configuration parameters.
         *
         * Implementations must throw std::invalid_argument on invalid configuration.
         */
        void validate() const
        {
            if ( k_ == 0 )
            {
                throw std::invalid_argument( "TopKConfig: k must be greater than zero" );
            }
        }

    private:
        size_t k_; ///< The number of top tokens to consider during decoding.
    };
}
