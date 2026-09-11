/**
 * @file AttentionOutputGate.Config.ixx
 * @brief Configuration for the attention output-gate component.
 *
 * The gate has no structurally required parameters: its width comes from the input shape at
 * build time and its activation is a compile-time template parameter, the same shape Swiglu
 * carries for its gate function. The default constructor is correct.
 */

module;
#include <string>
#include <sstream>

export module Dnn.Components.AttentionOutputGateConfig;

import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    export class AttentionOutputGateConfig : public ComponentConfig
    {
    public:

        AttentionOutputGateConfig() = default;

        void validate() const override
        {}

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            return meta;
        }

        void fromMetadata( const SerializationMetadata& /*meta*/ ) override
        {
            // Carries no serialized state -- the gate width is derived from the attention
            // geometry of the surrounding block, not stored on the config.
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "AttentionOutputGateConfig( )";

            return oss.str();
        }
    };
}
