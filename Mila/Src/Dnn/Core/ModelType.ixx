/**
 * @file ModelType.ixx
 * @brief Architecture identity for top-level model networks.
 *
 * ModelType names the model *family* (GPT-2, LLaMA, Gemma, ...). It is a distinct
 * axis from ComponentType, which names the structural *kind* of a component
 * (Linear, Transformer block, Network). A LlamaTransformer and a GemmaTransformer
 * are both ComponentType::Network; they differ by ModelType. Keeping the two axes
 * separate is why the per-architecture values were removed from ComponentType.
 */

module;
#include <string>
#include <string_view>
#include <cctype>

export module Dnn.ModelType;

namespace Mila::Dnn
{
    /**
     * @enum ModelType
     * @brief Canonical list of framework-known model architectures.
     */
    export enum class ModelType : int
    {
        Unknown = 0,

        Gpt2,       ///< GPT-2 style decoder network
        Llama,      ///< LLaMA 3 style decoder network
        Gemma,      ///< Gemma 4 style decoder network
        Mistral,    ///< Mistral style decoder network
        Bert,       ///< BERT style encoder network
    };

    /**
     * @brief Convert a ModelType to its canonical architecture name.
     *
     * Overloads (does not collide with) the ComponentType toString: the argument
     * type disambiguates. Returns "Unknown" for unrecognized values.
     */
    export inline std::string toString( ModelType t ) noexcept
    {
        switch ( t )
        {
            case ModelType::Gpt2:
                return "Gpt2";
            case ModelType::Llama:
                return "Llama";
            case ModelType::Gemma:
                return "Gemma";
            case ModelType::Mistral:
                return "Mistral";
            case ModelType::Bert:
                return "Bert";

            default:
                return "Unknown";
        }
    }

    /**
     * @brief Parse a case-insensitive architecture name into a ModelType.
     *
     * Named distinctly from the ComponentType parser: overloading on return type
     * alone is not possible, and both parsers take a string_view. Returns
     * ModelType::Unknown for unrecognized input.
     */
    export inline ModelType modelTypeFromString( std::string_view s ) noexcept
    {
        std::string low; low.reserve( s.size() );
        for ( unsigned char c : s ) low.push_back( static_cast<char>(std::tolower( c )) );

        if ( low == "gpt2" )
            return ModelType::Gpt2;
        if ( low == "llama" )
            return ModelType::Llama;
        if ( low == "gemma" )
            return ModelType::Gemma;
        if ( low == "mistral" )
            return ModelType::Mistral;
        if ( low == "bert" )
            return ModelType::Bert;

        return ModelType::Unknown;
    }
}
