/**
 * @file ComponentType.ixx
 * @brief Enumeration of built-in component types supported by the deserializer.
 *
 * Provides an enum and helper conversion utilities used by serialization,
 * factories and concise diagnostics.
 */

module;
#include <string>
#include <string_view>
#include <algorithm>
#include <cctype>

export module Dnn.ComponentType;

namespace Mila::Dnn
{
    /**
     * @enum ComponentType
     * @brief Canonical list of framework-known component types.
     *
     * These values are used by the deserializer and factory code to identify
     * component implementations. Values 1..999 are reserved for built-in
     * components; values >= CustomComponentStart are available for user
     * defined components or extensions.
     */
    export enum class ComponentType : int
    {
        Unknown = 0,

        // Built-in component types (1-999)

        // Leaf components
        Linear,
        Gelu,
        Swiglu,
        LayerNorm,
        RmsNorm,
        Softmax,
        MultiHeadAttention,
        GroupedQueryAttention,
        Residual,
        Lpe,
        Rope,

        // Composite components
        Mlp,
        Transformer,

        // Top-level networks
        Network,

        // REVIEW: Do we need Gpt2, Llama, etc.
        Gpt2,       ///< GPT-2 style transformer network
        Llama,      ///< LLaMA style transformer network
        Mistral,    ///< Mistral style transformer network
        Bert,       ///< BERT style transformer network

        // Reserve range for custom components (1000+)
        CustomComponentStart = 1000,

        MockComponent = CustomComponentStart,  ///< Example custom component for testing
    };

    /**
     * @brief Convert a ComponentType enum value to its canonical name.
     *
     * Returns a human-readable name suitable for logs and metadata fields
     * (for example "Linear", "Transformer"). Always returns "Unknown" for
     * unrecognized enum values.
     *
     * @param t ComponentType enum value
     * @return std::string Canonical name for the component type
     */
    export inline std::string toString( ComponentType t ) noexcept
    {
        switch ( t )
        {
            case ComponentType::Linear:
                return "Linear";
            case ComponentType::Gelu:
                return "Gelu";
            case ComponentType::Swiglu:
                return "Swiglu";
            case ComponentType::LayerNorm:
                return "LayerNorm";
            case ComponentType::RmsNorm:
                return "RmsNorm";
            case ComponentType::Softmax:
                return "Softmax";
            case ComponentType::MultiHeadAttention:
                return "MultiHeadAttention";
            case ComponentType::Residual:
                return "Residual";
            case ComponentType::Mlp:
                return "MLP";
            case ComponentType::Transformer:
                return "Transformer";
            case ComponentType::Lpe:
                return "Lpe";
            case ComponentType::Rope:
                return "Rope";
            
            case ComponentType::Network:
                return "Network";

            default:
                return "Unknown";
        }
    }

    /**
     * @brief Parse a case-insensitive component name into a ComponentType.
     *
     * Accepts canonical names (case-insensitive) produced by `toString` and
     * returns the corresponding enum value. Returns `ComponentType::Unknown`
     * if the input does not match any known type.
     *
     * @param s Input string (case-insensitive)
     * @return ComponentType Matching enum value or `Unknown`
     */
    export inline ComponentType fromString( std::string_view s ) noexcept
    {
        std::string low; low.reserve( s.size() );
        for ( unsigned char c : std::string_view( s ) ) low.push_back( static_cast<char>(std::tolower( c )) );

        if ( low == "linear" )
            return ComponentType::Linear;
        if ( low == "gelu" )
            return ComponentType::Gelu;
        if ( low == "swiglu" )
            return ComponentType::Swiglu;
        if ( low == "layernorm" )
            return ComponentType::LayerNorm;
        if ( low == "rmsnorm" )
            return ComponentType::RmsNorm;
        if ( low == "softmax" )
            return ComponentType::Softmax;
        if ( low == "attention" )
            return ComponentType::MultiHeadAttention;
        if ( low == "residual" )
            return ComponentType::Residual;
        if ( low == "mlp" )
            return ComponentType::Mlp;
        if ( low == "transformer" )
            return ComponentType::Transformer;
        if ( low == "lpe" )
            return ComponentType::Lpe;
        if ( low == "rope" )
            return ComponentType::Rope;
        if ( low == "network" )
            return ComponentType::Network;

        return ComponentType::Unknown;
    }

    /**
     * @brief Get the short 2..4 character type identifier for a ComponentType.
     *
     * The short type id is intended for compact labels in serialized metadata
     * and concise diagnostics (examples: "fc" for Linear, "mlp" for MLP,
     * "tf" for Transformer). Returns "Unknown" for unrecognized types.
     *
     * @param t ComponentType enum value
     * @return std::string Short lowercase identifier (2..4 chars) or "Unknown"
     */
    export inline std::string toTypeId( ComponentType t ) noexcept
    {
        switch ( t )
        {
            case ComponentType::Linear:
                return "fc";
            case ComponentType::Gelu:
                return "gelu";
            case ComponentType::Swiglu:
                return "swig";
            case ComponentType::LayerNorm:
                return "ln";
            case ComponentType::RmsNorm:
                return "rmsn";
            case ComponentType::Softmax:
                return "smax";
            case ComponentType::MultiHeadAttention:
                return "mha";
            case ComponentType::Residual:
                return "res";
            case ComponentType::Mlp:
                return "mlp";
            case ComponentType::Transformer:
                return "tf";
            case ComponentType::Lpe:
                return "lpe";
            case ComponentType::Rope:
                return "rope";
            case ComponentType::Network:
                return "net";

            default:
                return "Unknown";
        }
    }

    /**
     * @brief Map a short type identifier back to a ComponentType enum.
     *
     * Accepts the compact lowercase identifiers produced by `toTypeId` and
     * returns the corresponding enum value. Returns `ComponentType::Unknown`
     * for unrecognized identifiers.
     *
     * @param s Short lowercase type id (for example "fc", "mlp", "tf")
     * @return ComponentType Matching enum value or `Unknown`
     */
    export inline ComponentType fromTypeId( std::string_view s ) noexcept
    {
        if ( s == "fc" )
            return ComponentType::Linear;
        if ( s == "gelu" )
            return ComponentType::Gelu;
        if ( s == "swig" )
            return ComponentType::Swiglu;
        if ( s == "ln" )
            return ComponentType::LayerNorm;
        if ( s == "rmsn" )
            return ComponentType::RmsNorm;
        if ( s == "smax" )
            return ComponentType::Softmax;
        if ( s == "mha" )
            return ComponentType::MultiHeadAttention;
        if ( s == "res" )
            return ComponentType::Residual;
        if ( s == "mlp" )
            return ComponentType::Mlp;
        if ( s == "tf" )
            return ComponentType::Transformer;
        if ( s == "lpe" )
            return ComponentType::Lpe;
        if ( s == "rope" )
            return ComponentType::Rope;
        if ( s == "net" )
            return ComponentType::Network;
        
        return ComponentType::Unknown;
    }
}